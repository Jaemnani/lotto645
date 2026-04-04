import { useState } from 'react'
import { Dices, Save, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import LottoBall from './LottoBall'
import { api } from '../api'
import type { ExtractResponse } from '../types'
import { STRATEGY_DESC, STRATEGY_LABELS } from '../types'

export default function ExtractPanel() {
  const [ballSet, setBallSet] = useState('0')
  const [strategy, setStrategy] = useState('1')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState<ExtractResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleExtract() {
    setLoading(true)
    setError(null)
    try {
      const res = await api.extract(Number(ballSet), Number(strategy), false)
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      const res = await api.extract(Number(ballSet), Number(strategy), true)
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <Dices className="w-5 h-5 text-primary" />
          번호 추출
        </CardTitle>
        <CardDescription>
          LSTM 모델이 예측한 번호를 추출합니다
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* 옵션 선택 */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-muted-foreground">공세트</label>
            <Select value={ballSet} onValueChange={(v) => { if (v !== null) setBallSet(v) }}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">자동 선택</SelectItem>
                {[1, 2, 3, 4, 5].map((n) => (
                  <SelectItem key={n} value={String(n)}>
                    세트 {n}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-muted-foreground">전략</label>
            <Select value={strategy} onValueChange={(v) => { if (v !== null) setStrategy(v) }}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[1, 2, 3, 4].map((n) => (
                  <SelectItem key={n} value={String(n)}>
                    {STRATEGY_LABELS[n]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* 전략 설명 */}
        <p className="text-xs text-muted-foreground bg-muted/50 rounded-md px-3 py-2">
          {STRATEGY_DESC[Number(strategy)]}
        </p>

        {/* 버튼 */}
        <div className="flex gap-2">
          <Button onClick={handleExtract} disabled={loading} className="flex-1">
            {loading
              ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />추출 중...</>
              : <><Dices className="w-4 h-4 mr-2" />번호 추출</>
            }
          </Button>
          {result && !result.saved && (
            <Button variant="outline" onClick={handleSave} disabled={saving}>
              {saving
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : <><Save className="w-4 h-4 mr-1" />저장</>
              }
            </Button>
          )}
        </div>

        {/* 오류 */}
        {error && (
          <p className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">{error}</p>
        )}

        {/* 로딩 스켈레톤 */}
        {loading && (
          <div className="space-y-3">
            <div className="flex gap-2 justify-center">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="w-11 h-11 rounded-full" />
              ))}
            </div>
          </div>
        )}

        {/* 결과 */}
        {result && !loading && (
          <div className="space-y-3 pt-1">
            <div className="flex gap-2 justify-center flex-wrap">
              {result.numbers.map((n) => (
                <LottoBall key={n} number={n} size="lg" />
              ))}
            </div>
            <div className="flex items-center justify-center gap-2 flex-wrap">
              <Badge variant="secondary">세트 {result.ball_set}</Badge>
              <Badge variant="secondary">{STRATEGY_LABELS[result.strategy]}</Badge>
              <Badge variant="outline">{result.target_round}회차 응모용</Badge>
              {result.saved && (
                <Badge className="bg-green-600 text-white">저장 완료</Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
