/** Local-only portrait history. No portrait bytes are sent anywhere except to
 * the existing /api/jobs endpoint when the user requests generation. */
export interface RecentPortrait {
  id: string
  name: string
  type: string
  file: Blob
  savedAt: number
  sampleName: string | null
  jobId: string | null
  method: string | null
  vectorMode: 'contour' | 'centerline' | null
  size: number | null
}

const DATABASE = 'robotportrait-local-history'
const STORE = 'portraits'
export const MAX_RECENT_PORTRAITS = 6

function requestPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'))
  })
}

function openHistory(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('Browser storage is unavailable'))
      return
    }
    const request = indexedDB.open(DATABASE, 1)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Cannot open browser storage'))
    request.onblocked = () => reject(new Error('Browser storage is blocked'))
  })
}

async function withStore<T>(
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openHistory()
  try {
    const transaction = db.transaction(STORE, mode)
    const result = requestPromise(action(transaction.objectStore(STORE)))
    // Wait until the write transaction is committed before reporting success.
    const done = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve()
      transaction.onabort = () => reject(transaction.error ?? new Error('Storage transaction aborted'))
      transaction.onerror = () => reject(transaction.error ?? new Error('Storage transaction failed'))
    })
    const value = await result
    await done
    return value
  } finally {
    db.close()
  }
}

export async function listRecentPortraits(): Promise<RecentPortrait[]> {
  const records = await withStore('readonly', store => store.getAll()) as RecentPortrait[]
  return records.sort((a, b) => b.savedAt - a.savedAt)
}

export async function saveRecentPortrait(record: RecentPortrait): Promise<void> {
  await withStore('readwrite', store => store.put(record))
  const records = await listRecentPortraits()
  for (const old of records.slice(MAX_RECENT_PORTRAITS)) {
    await removeRecentPortrait(old.id)
  }
}

export async function removeRecentPortrait(id: string): Promise<void> {
  await withStore('readwrite', store => store.delete(id))
}

export async function clearRecentPortraits(): Promise<void> {
  await withStore('readwrite', store => store.clear())
}
