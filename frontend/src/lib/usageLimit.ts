/**
 * Daily usage limit tracking via localStorage.
 * Limits each user to a configurable number of image generations per day.
 */

const STORAGE_KEY_PREFIX = "anglecam_usage_";
const DAILY_LIMIT = 5;

function getTodayKey(): string {
  const today = new Date();
  const dateStr =
    today.getFullYear() +
    "-" +
    String(today.getMonth() + 1).padStart(2, "0") +
    "-" +
    String(today.getDate()).padStart(2, "0");
  return STORAGE_KEY_PREFIX + dateStr;
}

/** Get the number of generations used today */
export function getUsedToday(): number {
  try {
    const key = getTodayKey();
    const stored = localStorage.getItem(key);
    if (stored === null) return 0;
    const count = parseInt(stored, 10);
    return isNaN(count) ? 0 : count;
  } catch {
    return 0;
  }
}

/** Get the number of remaining generations for today */
export function getRemainingToday(): number {
  return Math.max(0, DAILY_LIMIT - getUsedToday());
}

/** Get the daily limit */
export function getDailyLimit(): number {
  return DAILY_LIMIT;
}

/** Check if the user can generate more images today */
export function canGenerate(): boolean {
  return getUsedToday() < DAILY_LIMIT;
}

/** Increment the usage count for today */
export function incrementUsage(): void {
  try {
    const key = getTodayKey();
    const current = getUsedToday();
    localStorage.setItem(key, String(current + 1));
    // Clean up old entries (keep only last 7 days)
    cleanupOldEntries();
  } catch {
    // localStorage might be unavailable
  }
}

/** Clean up usage entries older than 7 days */
function cleanupOldEntries(): void {
  try {
    const keysToRemove: string[] = [];
    const now = new Date();

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(STORAGE_KEY_PREFIX)) {
        const dateStr = key.substring(STORAGE_KEY_PREFIX.length);
        const entryDate = new Date(dateStr);
        if (!isNaN(entryDate.getTime())) {
          const diffDays =
            (now.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24);
          if (diffDays > 7) {
            keysToRemove.push(key);
          }
        }
      }
    }

    keysToRemove.forEach((key) => localStorage.removeItem(key));
  } catch {
    // ignore cleanup errors
  }
}
