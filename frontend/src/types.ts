export interface ExtractResponse {
  numbers: number[]
  ball_set: number
  strategy: number
  target_round: number
  saved: boolean
}

export interface DrawResult {
  round: number | null
  draw_date?: string
  numbers?: number[]
  bonus?: number
  message?: string
}

export interface AnnouncementStats {
  '1등': number
  '2등': number
  '3등': number
  '4등': number
  '5등': number
  '낙첨': number
  total: number
}

export interface Announcement {
  round: number | null
  draw_date?: string
  winning_numbers?: { numbers: number[]; bonus: number }
  stats?: AnnouncementStats
  total_extractions?: number
  published_at?: string
  message?: string
}

export const STRATEGY_LABELS: Record<number, string> = {
  1: '확률 상위',
  2: '가중 랜덤',
  3: '구간 균형',
  4: 'Cold 번호 포함',
}

export const STRATEGY_DESC: Record<number, string> = {
  1: '모델이 가장 높게 예측한 조합',
  2: '확률 기반 다양성 확보',
  3: '낮은-중간-높은 번호 골고루',
  4: '오래 안 나온 번호 포함',
}
