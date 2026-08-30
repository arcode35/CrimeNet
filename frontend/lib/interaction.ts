export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : Boolean(error && typeof error === "object" && "name" in error && error.name === "AbortError");
}

function abortError(): DOMException {
  return new DOMException("Superseded by a newer request", "AbortError");
}

/**
 * Owns one independently changing request stream. Aborting saves browser/network
 * work; the generation check also protects against request implementations that
 * resolve after ignoring an AbortSignal.
 */
export class LatestRequest {
  private generation = 0;
  private controller: AbortController | null = null;

  async run<T>(request: (signal: AbortSignal) => Promise<T>, upstream?: AbortSignal): Promise<T> {
    const requestGeneration = ++this.generation;
    this.controller?.abort();

    const controller = new AbortController();
    this.controller = controller;
    const abortFromUpstream = () => controller.abort(upstream?.reason);
    if (upstream?.aborted) abortFromUpstream();
    else upstream?.addEventListener("abort", abortFromUpstream, { once: true });

    try {
      const result = await request(controller.signal);
      if (controller.signal.aborted || requestGeneration !== this.generation) throw abortError();
      return result;
    } finally {
      upstream?.removeEventListener("abort", abortFromUpstream);
      if (this.controller === controller) this.controller = null;
    }
  }

  cancel(): void {
    this.generation += 1;
    this.controller?.abort();
    this.controller = null;
  }
}

export function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(abortError());
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export type ThrottledCommit<T> = {
  push(value: T): void;
  flush(value: T): void;
  cancel(): void;
};

/** Leading + trailing throttle. `flush` is used on slider release. */
export function createThrottledCommit<T>(
  commit: (value: T) => void,
  intervalMilliseconds = 200,
): ThrottledCommit<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: T | undefined;
  let hasPending = false;
  let lastCommittedAt = Number.NEGATIVE_INFINITY;
  let lastCommitted: T | undefined;
  let hasCommitted = false;

  const execute = () => {
    if (!hasPending) return;
    const value = pending as T;
    pending = undefined;
    hasPending = false;
    lastCommittedAt = Date.now();
    lastCommitted = value;
    hasCommitted = true;
    commit(value);
  };

  const clearTimer = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  };

  return {
    push(value) {
      pending = value;
      hasPending = true;
      const remaining = intervalMilliseconds - (Date.now() - lastCommittedAt);
      if (remaining <= 0) {
        clearTimer();
        execute();
      } else if (timer === null) {
        timer = setTimeout(() => {
          timer = null;
          execute();
        }, remaining);
      }
    },
    flush(value) {
      clearTimer();
      if (!hasPending && hasCommitted && Object.is(value, lastCommitted)) return;
      pending = value;
      hasPending = true;
      execute();
    },
    cancel() {
      clearTimer();
      pending = undefined;
      hasPending = false;
    },
  };
}
