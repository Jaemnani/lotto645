import { useEffect, useState } from 'react'
import { Megaphone, Trophy, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Separator } from '@/components/ui/separator'
import LottoBall from './LottoBall'
import { api } from '../api'
import type { Announcement } from '../types'

const RANK_COLORS: Record<string, string> = {
  '1등': 'bg-yellow-100 text-yellow-800 border-yellow-300',
  '2등': 'bg-orange-100 text-orange-800 border-orange-300',
  '3등': 'bg-blue-100 text-blue-800 border-blue-300',
  '4등': 'bg-green-100 text-green-800 border-green-300',
  '5등': 'bg-gray-100 text-gray-700 border-gray-300',
  '낙첨': 'bg-slate-50 text-slate-500 border-slate-200',
}

export default function AnnouncementPanel() {
  const [ann, setAnn] = useState<Announcement | null>(null)
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    try {
      setAnn(await api.latestAnnouncement())
    } catch {
      setAnn({ round: null, message: '공지사항을 불러오지 못했습니다' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-xl">
              <Megaphone className="w-5 h-5 text-primary" />
              주간 공지사항
            </CardTitle>
            <CardDescription>매주 토요일 추첨 후 통계 업데이트</CardDescription>
          </div>
          <button
            onClick={load}
            className="p-1.5 rounded-md hover:bg-muted transition-colors"
            title="새로고침"
          >
            <RefreshCw className={`w-4 h-4 text-muted-foreground ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </CardHeader>
      <CardContent>
        {loading && (
          <div className="space-y-3">
            <Skeleton className="h-4 w-32" />
            <div className="flex gap-2">
              {Array.from({ length: 7 }).map((_, i) => (
                <Skeleton key={i} className="w-10 h-10 rounded-full" />
              ))}
            </div>
            <Skeleton className="h-20 w-full rounded-lg" />
          </div>
        )}

        {!loading && ann?.message && !ann.round && (
          <div className="text-center py-8 text-muted-foreground">
            <Trophy className="w-10 h-10 mx-auto mb-2 opacity-30" />
            <p className="text-sm">{ann.message}</p>
            <p className="text-xs mt-1">토요일 추첨 후 통계가 업데이트됩니다</p>
          </div>
        )}

        {!loading && ann?.round && ann.winning_numbers && ann.stats && (
          <div className="space-y-4">
            {/* 회차 정보 */}
            <div className="flex items-center gap-2 flex-wrap">
              <Badge className="text-sm px-3 py-1">{ann.round}회차</Badge>
              <span className="text-sm text-muted-foreground">{ann.draw_date}</span>
            </div>

            {/* 당첨 번호 */}
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">당첨 번호</p>
              <div className="flex gap-2 items-center flex-wrap">
                {ann.winning_numbers.numbers.map((n) => (
                  <LottoBall key={n} number={n} size="md" />
                ))}
                <span className="text-muted-foreground mx-1">+</span>
                <LottoBall number={ann.winning_numbers.bonus} size="md" isBonus />
              </div>
            </div>

            <Separator />

            {/* 통계 */}
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">
                등수별 통계 <span className="font-normal">(총 {ann.stats.total}개 추출)</span>
              </p>
              <div className="grid grid-cols-3 gap-2">
                {(['1등', '2등', '3등', '4등', '5등', '낙첨'] as const).map((rank) => (
                  <div
                    key={rank}
                    className={`rounded-lg border px-3 py-2 text-center ${RANK_COLORS[rank]}`}
                  >
                    <p className="text-xs font-medium">{rank}</p>
                    <p className="text-lg font-bold">{ann.stats![rank]}</p>
                    <p className="text-xs opacity-70">
                      {ann.stats!.total > 0
                        ? `${((ann.stats![rank] / ann.stats!.total) * 100).toFixed(1)}%`
                        : '0%'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
