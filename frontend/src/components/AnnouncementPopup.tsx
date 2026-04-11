import { useEffect, useState } from 'react'
import { Trophy, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import LottoBall from './LottoBall'
import { api } from '../api'
import type { Announcement } from '../types'

const STORAGE_KEY = 'lastSeenAnnouncementRound'

const RANK_COLORS: Record<string, string> = {
  '1등': 'bg-yellow-100 text-yellow-800 border-yellow-300',
  '2등': 'bg-orange-100 text-orange-800 border-orange-300',
  '3등': 'bg-blue-100 text-blue-800 border-blue-300',
  '4등': 'bg-green-100 text-green-800 border-green-300',
  '5등': 'bg-gray-100 text-gray-700 border-gray-300',
  '낙첨': 'bg-slate-50 text-slate-500 border-slate-200',
}

export default function AnnouncementPopup() {
  const [ann, setAnn] = useState<Announcement | null>(null)

  useEffect(() => {
    api.latestAnnouncement().then((data) => {
      if (!data.round) return
      const lastSeen = Number(localStorage.getItem(STORAGE_KEY) ?? 0)
      if (data.round > lastSeen) {
        setAnn(data)
      }
    }).catch(() => {})
  }, [])

  function dismiss() {
    if (ann?.round) {
      localStorage.setItem(STORAGE_KEY, String(ann.round))
    }
    setAnn(null)
  }

  if (!ann?.round || !ann.winning_numbers || !ann.stats) return null

  return (
    /* 배경 오버레이 */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
      onClick={dismiss}
    >
      {/* 팝업 카드 */}
      <div
        className="bg-card w-full max-w-sm rounded-2xl shadow-2xl p-6 relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 닫기 버튼 */}
        <button
          onClick={dismiss}
          className="absolute top-4 right-4 p-1 rounded-full hover:bg-muted transition-colors"
        >
          <X className="w-4 h-4 text-muted-foreground" />
        </button>

        {/* 타이틀 */}
        <div className="flex items-center gap-2 mb-4">
          <Trophy className="w-5 h-5 text-yellow-500" />
          <h2 className="text-lg font-bold">추첨 결과 발표!</h2>
        </div>

        {/* 회차 / 날짜 */}
        <div className="flex items-center gap-2 mb-3">
          <Badge className="text-sm px-3 py-1">{ann.round}회차</Badge>
          <span className="text-sm text-muted-foreground">{ann.draw_date}</span>
        </div>

        {/* 당첨 번호 */}
        <p className="text-xs font-medium text-muted-foreground mb-2">당첨 번호</p>
        <div className="flex gap-1.5 items-center flex-wrap mb-4">
          {ann.winning_numbers.numbers.map((n) => (
            <LottoBall key={n} number={n} size="sm" />
          ))}
          <span className="text-muted-foreground mx-0.5">+</span>
          <LottoBall number={ann.winning_numbers.bonus} size="sm" isBonus />
        </div>

        <Separator className="mb-4" />

        {/* 통계 */}
        <p className="text-xs font-medium text-muted-foreground mb-2">
          우리 서비스 추출 결과 <span className="font-normal">(총 {ann.stats.total}개)</span>
        </p>
        <div className="grid grid-cols-3 gap-1.5">
          {(['1등', '2등', '3등', '4등', '5등', '낙첨'] as const).map((rank) => (
            <div
              key={rank}
              className={`rounded-lg border px-2 py-1.5 text-center ${RANK_COLORS[rank]}`}
            >
              <p className="text-xs font-medium">{rank}</p>
              <p className="text-base font-bold">{ann.stats![rank]}</p>
            </div>
          ))}
        </div>

        {/* 확인 버튼 */}
        <button
          onClick={dismiss}
          className="mt-5 w-full bg-primary text-primary-foreground rounded-xl py-2.5 text-sm font-medium hover:opacity-90 transition-opacity"
        >
          확인
        </button>
      </div>
    </div>
  )
}
