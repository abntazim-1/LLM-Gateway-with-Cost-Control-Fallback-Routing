import asyncio
import logging
from typing import Dict, Any, Optional
from gateway.ledger.base_store import BaseLedgerStore

logger = logging.getLogger(__name__)

class AsyncLedgerQueue:
    """
    Non-blocking asynchronous queue wrapper for recording spend ledger entries.
    Offloads synchronous/database IO away from HTTP request loops into background task workers.
    """

    def __init__(self, store: BaseLedgerStore, maxsize: int = 10000):
        self.store = store
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        """Start the background consumer worker task."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Flush remaining queued items and stop the worker task."""
        self._running = False
        if self._worker_task:
            # Signal worker to finish queue
            await self.queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def record_request(
        self, 
        api_key: str, 
        req_id: str, 
        backend: str, 
        model: str, 
        prompt_tokens: int, 
        comp_tokens: int, 
        cost: float, 
        latency: float
    ):
        """Enqueue request payload for background processing without blocking."""
        item = {
            "api_key": api_key,
            "req_id": req_id,
            "backend": backend,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "comp_tokens": comp_tokens,
            "cost": cost,
            "latency": latency
        }
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.error("AsyncLedgerQueue full! Fallback to direct async write.")
            await self.store.record_request(**item)

    async def _worker_loop(self):
        """Worker loop draining the queue and persisting entries."""
        while self._running or not self.queue.empty():
            try:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    await self.store.record_request(
                        api_key=item["api_key"],
                        req_id=item["req_id"],
                        backend=item["backend"],
                        model=item["model"],
                        prompt_tokens=item["prompt_tokens"],
                        comp_tokens=item["comp_tokens"],
                        cost=item["cost"],
                        latency=item["latency"]
                    )
                except Exception as e:
                    logger.error(f"Error persisting ledger queue item: {e}")
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AsyncLedgerQueue worker unexpected error: {e}")
