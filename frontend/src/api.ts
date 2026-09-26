import type { Announcement, DrawResult, ExtractResponse, ModelKey } from './types'

const BASE = import.meta.env.VITE_API_URL ?? ''

async function post<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: 'include' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  extract: (ball_set: number, strategy: number, model: ModelKey, save: boolean) =>
    post<ExtractResponse>('/api/extract', { ball_set, strategy, model, save }),

  latestDraw: () => get<DrawResult>('/api/draw/latest'),

  latestAnnouncement: () => get<Announcement>('/api/announcement/latest'),

  announcements: (limit = 5) =>
    get<Announcement[]>(`/api/announcements?limit=${limit}`),
}
