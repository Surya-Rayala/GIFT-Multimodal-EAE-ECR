"""UDP impairment proxy — simulates a bad network between a stream sender and the EAE.

    sender ──udp──► proxy (--listen) ──udp──► EAE (--forward)

Impairments (all optional, combinable, seeded for repeatability):

  --loss P            drop each datagram with probability P (0..1)
  --burst-loss P:L    start a loss burst with probability P per datagram,
                      dropping the next L datagrams (Wi-Fi style bursts)
  --outage A:D,...    drop everything from second A for D seconds
                      (seconds since the first datagram)
  --jitter-ms J       queueing jitter: each datagram is delayed 0..J ms but
                      delivery order is preserved (FIFO), as on a real link
  --reorder P         with probability P deliver a datagram --reorder-ms
                      late, i.e. after later ones (genuine reordering)
  --dup P             duplicate each datagram with probability P
  --rate-kbps R       token-bucket bandwidth cap; excess queues, and the
                      queue is tail-dropped beyond --queue-ms

Exits after --idle-exit seconds without traffic once traffic started, and
writes counters to --stats (JSON).
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import socket
import time
from typing import List, Tuple


def _addr(s: str) -> Tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)


def _windows(spec: str) -> List[Tuple[float, float]]:
    out = []
    for part in (spec or "").split(","):
        if part.strip():
            a, d = part.split(":")
            out.append((float(a), float(a) + float(d)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True, help="host:port to receive on")
    ap.add_argument("--forward", required=True, help="host:port to send to")
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--burst-loss", default="", help="P:L")
    ap.add_argument("--outage", default="", help="start:dur[,start:dur]")
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--reorder", type=float, default=0.0)
    ap.add_argument("--reorder-ms", type=float, default=5.0)
    ap.add_argument("--dup", type=float, default=0.0)
    ap.add_argument("--rate-kbps", type=float, default=0.0)
    ap.add_argument("--queue-ms", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--idle-exit", type=float, default=8.0)
    ap.add_argument("--stats", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outages = _windows(args.outage)
    burst_p, burst_len = 0.0, 0
    if args.burst_loss:
        p, n = args.burst_loss.split(":")
        burst_p, burst_len = float(p), int(n)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass
    rx.bind(_addr(args.listen))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = _addr(args.forward)

    stats = {"received": 0, "forwarded": 0, "dropped_random": 0, "dropped_burst": 0,
             "dropped_outage": 0, "dropped_queue": 0, "duplicated": 0, "reordered": 0}
    heap: List[Tuple[float, int, bytes]] = []
    seq = 0
    burst_left = 0
    t0 = None
    last_rx = None
    last_sent_seq = -1
    last_due = 0.0
    # token bucket
    rate_bps = args.rate_kbps * 1000.0
    link_free_at = 0.0

    while True:
        now = time.monotonic()
        timeout = 0.05
        if heap:
            timeout = max(0.0, min(timeout, heap[0][0] - now))
        rx.settimeout(timeout if timeout > 0 else 0.0005)
        try:
            data, _ = rx.recvfrom(65536)
        except (socket.timeout, BlockingIOError):
            data = None
        now = time.monotonic()
        if data is not None:
            if t0 is None:
                t0 = now
            last_rx = now
            stats["received"] += 1
            rel = now - t0
            drop = None
            if any(a <= rel < b for a, b in outages):
                drop = "dropped_outage"
            elif burst_left > 0:
                burst_left -= 1
                drop = "dropped_burst"
            elif burst_p and rng.random() < burst_p:
                burst_left = burst_len - 1
                drop = "dropped_burst"
            elif args.loss and rng.random() < args.loss:
                drop = "dropped_random"
            if drop:
                stats[drop] += 1
            else:
                due = now + (rng.random() * args.jitter_ms / 1000.0 if args.jitter_ms else 0.0)
                due = max(due, last_due)  # a queue delays but does not reorder
                last_due = due
                if args.reorder and rng.random() < args.reorder:
                    due += args.reorder_ms / 1000.0  # this one overtaken by later ones
                if rate_bps > 0:
                    start = max(due, link_free_at)
                    if start - now > args.queue_ms / 1000.0:
                        stats["dropped_queue"] += 1
                        continue
                    link_free_at = start + len(data) * 8.0 / rate_bps
                    due = link_free_at
                heapq.heappush(heap, (due, seq, data))
                seq += 1
                if args.dup and rng.random() < args.dup:
                    heapq.heappush(heap, (due + 0.0005, seq, data))
                    seq += 1
                    stats["duplicated"] += 1
        while heap and heap[0][0] <= now:
            _, s, payload = heapq.heappop(heap)
            if s < last_sent_seq:
                stats["reordered"] += 1
            last_sent_seq = max(last_sent_seq, s)
            try:
                tx.sendto(payload, dst)
                stats["forwarded"] += 1
            except OSError:
                pass
        if last_rx is not None and not heap and now - last_rx > args.idle_exit:
            break

    if args.stats:
        with open(args.stats, "w") as fh:
            json.dump(stats, fh, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
