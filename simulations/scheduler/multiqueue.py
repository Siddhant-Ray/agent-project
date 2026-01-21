import asyncio
import random
import time
from dataclasses import dataclass

# ----------------------------
# Parameters
# ----------------------------
SIM_TIME = 40

ARRIVAL_INTERVAL = 0.6        # moderate load
DEMOTION_T1 = 10              # Q1 → Q2
DEMOTION_T2 = 20              # Q2 → Q3

# High variance service times
SERVICE_TIME_MIN = 0.2
SERVICE_TIME_MAX = 4.0

# ----------------------------
# Request model
# ----------------------------
@dataclass
class Request:
    rid: int
    arrival_time: float
    expiry_time: float
    service_time: float     # ← fixed per request

    def valid(self, now):
        return now < self.expiry_time


# ----------------------------
# Queues (Q1 > Q2 > Q3)
# ----------------------------
q1 = asyncio.Queue()
q2 = asyncio.Queue()
q3 = asyncio.Queue()


def log(start, msg):
    now = time.monotonic() - start
    print(f"[{now:6.2f}] {msg}")


# ----------------------------
# Request generator
# ----------------------------
async def request_source(start):
    rid = 0
    while True:
        await asyncio.sleep(ARRIVAL_INTERVAL)
        now = time.monotonic() - start

        service_time = random.uniform(SERVICE_TIME_MIN, SERVICE_TIME_MAX)

        req = Request(
            rid=rid,
            arrival_time=now,
            expiry_time=now + 120,
            service_time=service_time,
        )

        await q1.put(req)
        log(start, f"request {rid} arrives")

        rid += 1


# ----------------------------
# Time-based demotion
# ----------------------------
async def demoter(start):
    demoted_q1 = set()
    demoted_q2 = set()

    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic() - start

        # Q1 → Q2
        if now >= DEMOTION_T1:
            tmp = []
            while not q1.empty():
                r = await q1.get()
                if r.valid(now) and r.rid not in demoted_q1:
                    demoted_q1.add(r.rid)
                    await q2.put(r)
                    log(start, f"request {r.rid} moves to Q2")
                else:
                    tmp.append(r)
            for r in tmp:
                await q1.put(r)

        # Q2 → Q3
        if now >= DEMOTION_T2:
            tmp = []
            while not q2.empty():
                r = await q2.get()
                if r.valid(now) and r.rid not in demoted_q2:
                    demoted_q2.add(r.rid)
                    await q3.put(r)
                    log(start, f"request {r.rid} moves to Q3")
                else:
                    tmp.append(r)
            for r in tmp:
                await q2.put(r)


# ----------------------------
# Strict-priority worker
# ----------------------------
async def priority_worker(start):
    while True:
        if not q1.empty():
            r = await q1.get()
        elif not q2.empty():
            r = await q2.get()
        elif not q3.empty():
            r = await q3.get()
        else:
            await asyncio.sleep(0.01)
            continue

        now = time.monotonic() - start
        if not r.valid(now):
            continue

        # Variable-length service → overtaking
        await asyncio.sleep(r.service_time)
        log(start, f"request {r.rid} finished")


# ----------------------------
# Main
# ----------------------------
async def main():
    start = time.monotonic()

    tasks = [
        asyncio.create_task(request_source(start)),
        asyncio.create_task(demoter(start)),
        asyncio.create_task(priority_worker(start)),
    ]

    await asyncio.sleep(SIM_TIME)
    for t in tasks:
        t.cancel()


asyncio.run(main())
