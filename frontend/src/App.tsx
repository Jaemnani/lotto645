import { useEffect, useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import ExtractPanel from './components/ExtractPanel'
import AnnouncementPanel from './components/AnnouncementPanel'
import AnnouncementPopup from './components/AnnouncementPopup'
import { api } from './api'
import './index.css'

function App() {
  const [nextRound, setNextRound] = useState<number | null>(null)

  useEffect(() => {
    api.latestDraw()
      .then((d) => { if (d.round) setNextRound(d.round + 1) })
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <AnnouncementPopup />
      {/* 헤더 */}
      <header className="border-b bg-card sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">🍀 로또 번호 추출기</h1>
            <p className="text-xs text-muted-foreground mt-0.5">LSTM 모델 기반 번호 추천 서비스</p>
          </div>
          {nextRound && (
            <Badge variant="outline" className="text-sm px-3 py-1">
              {nextRound}회차 응모 중
            </Badge>
          )}
        </div>
      </header>

      {/* 메인 */}
      <main className="max-w-2xl mx-auto px-4 py-6">
        <Tabs defaultValue="extract">
          <TabsList className="w-full mb-5">
            <TabsTrigger value="extract" className="flex-1">번호 추출</TabsTrigger>
            <TabsTrigger value="announcement" className="flex-1">공지사항</TabsTrigger>
          </TabsList>

          <TabsContent value="extract">
            <ExtractPanel />
          </TabsContent>

          <TabsContent value="announcement">
            <AnnouncementPanel />
          </TabsContent>
        </Tabs>
      </main>

      {/* 푸터 */}
      <footer className="border-t mt-10">
        <p className="text-center text-xs text-muted-foreground py-4">
          매주 토요일 21시 추첨 결과 반영 · 이 서비스는 참고용입니다
        </p>
      </footer>
    </div>
  )
}

export default App
