import { afterEach, describe, expect, it, vi } from "vitest";
import { createThrottledCommit, isAbortError, LatestRequest } from "@/lib/interaction";

afterEach(() => vi.useRealTimers());

describe("latest-request-wins interaction control", () => {
  it("aborts the previous request and rejects a stale response that ignores cancellation", async () => {
    const gate = new LatestRequest();
    let resolveFirst!: (value: string) => void;
    let firstSignal: AbortSignal | undefined;
    const first = gate.run(
      (signal) =>
        new Promise<string>((resolve) => {
          firstSignal = signal;
          resolveFirst = resolve;
        }),
    );
    const second = gate.run(async () => "newest");

    expect(firstSignal?.aborted).toBe(true);
    resolveFirst("stale");
    await expect(second).resolves.toBe("newest");
    await expect(first).rejects.toSatisfy(isAbortError);
  });

  it("throttles drag updates and always commits the final slider value", () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000);
    const committed: number[] = [];
    const throttle = createThrottledCommit<number>((value) => committed.push(value), 200);

    throttle.push(1);
    throttle.push(2);
    throttle.push(3);
    expect(committed).toEqual([1]);

    vi.advanceTimersByTime(100);
    throttle.flush(4);
    expect(committed).toEqual([1, 4]);
    vi.advanceTimersByTime(500);
    expect(committed).toEqual([1, 4]);
  });
});
