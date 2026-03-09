import asyncio
import concurrent.futures
from typing import Callable, List, Any, TypeVar
from tqdm.asyncio import tqdm as async_tqdm

T = TypeVar("T")
R = TypeVar("R")


class AsyncBatchProcessor:
    """
    A reusable async batch processor with semaphore-based concurrency control
    and exponential backoff retries.
    """

    def __init__(
        self,
        max_concurrent: int = 36,
        max_retries: int = 3,
        initial_backoff: float = 2.0,
    ):
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    async def _process_item(
        self,
        item: T,
        process_fn: Callable[[T], R],
        semaphore: asyncio.Semaphore,
        executor: concurrent.futures.ThreadPoolExecutor,
        on_failure: Callable[[T, Exception], Any] | None = None,
    ) -> R | None:
        """Process a single item with retries."""
        async with semaphore:
            loop = asyncio.get_running_loop()

            for attempt in range(self.max_retries):
                try:
                    result = await loop.run_in_executor(executor, process_fn, item)
                    return result
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        sleep_time = self.initial_backoff * (2 ** attempt)
                        await asyncio.sleep(sleep_time)
                    else:
                        if on_failure:
                            on_failure(item, e)
                        return None

    async def run(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        on_failure: Callable[[T, Exception], Any] | None = None,
        desc: str = "Processing",
    ) -> List[R | None]:
        """
        Process a list of items concurrently.

        Args:
            items: List of items to process.
            process_fn: A synchronous function that processes a single item.
            on_failure: Optional callback invoked when all retries fail for an item.
            desc: Description for the progress bar.

        Returns:
            List of results (None for failed items).
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent + 5) as executor:
            tasks = [
                self._process_item(item, process_fn, semaphore, executor, on_failure)
                for item in items
            ]
            results = await async_tqdm.gather(*tasks, desc=desc)

        return results
