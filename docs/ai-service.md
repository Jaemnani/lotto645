# Workshop / AI Services 신규 프로젝트 등록 가이드

aib.vote 의 `/ai-services` 그리드에 새 GitHub 프로젝트를 노출시키기 위한 절차. 카드 / 상세 페이지가 비주얼이 빈약하지 않게 보이도록 (A) 대상 레포 측 README/메타데이터 정비와 (B) knowai-space 측 DB 등록 두 단계로 나누어 안내한다.

> **SOT 주의**: 이 가이드의 동작 근거는 [src/lib/github.ts](../src/lib/github.ts) 와 [src/app/api/admin/workshop/route.ts](../src/app/api/admin/workshop/route.ts). 자동 추출 헤더 이름이나 필수 필드 스키마가 바뀌면 본 문서도 동기화 PR.

---

## 전제 — 이것은 "라이브 웹 서비스 레지스트리" 가 아니다

`/ai-services` 는 **GitHub 프로젝트 쇼케이스(대시보드)** 다. 상세 페이지([\[slug\]/page.tsx](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx))가 렌더링하는 내용은 전부 **GitHub repo 메타데이터** — stars · contributors · open issues · releases · 언어 비율 · README Overview · roadmap 이슈. **배포된 웹 서비스(데모)는 부가 요소일 뿐, 등록의 필수 조건이 아니다.** 라이브러리/CLI/연구 코드처럼 "동작하는 사이트"가 없는 프로젝트도 그대로 등록된다.

### Hard requirements — 카드가 뜨기 위한 최소 조건 (이 2개면 충분)

1. **GitHub repo 가 public** — private 이면 상세 페이지가 GitHub fetch 실패로 자물쇠 fallback 화면만 표시 ([page.tsx:449-460](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L449)).
2. **`workshop_projects` 테이블에 `is_active=true` 행** — 리스트/상세 쿼리가 모두 `.eq('is_active', true)` 로 필터 ([queries/workshop.ts:17](../src/lib/supabase/queries/workshop.ts#L17)). POST 등록 핸들러가 자동으로 `true` 설정.

> 빌드 이후 추가된 slug 도 `dynamicParams = true` ([page.tsx:46](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L46)) 덕분에 on-demand 렌더 — 재빌드 없이 동작하며 사이트 다운 위험 없음.

그 외 (README 섹션, LICENSE, release, 데모 URL, 이미지) 는 전부 **카드/상세를 풍성하게 만드는 선택 요소**이며, 일부는 특정 UI(데모 버튼, Overview 블록)를 띄우는 트리거다 — 아래 섹션 0 표와 정정 사항 참고.

---

## 0) 사전 점검 체크리스트

대상 레포(`owner/repo`)에 대해 `gh api` 로 다음 항목을 확인. 충족 못한 항목은 섹션 A 에서 보강.

| 항목                  | 확인 명령                                                          | 등록 후 표시되는 모습                                                          |
| --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Visibility = PUBLIC   | `gh repo view OWNER/REPO --json visibility`                        | GitHub stats 정상 fetch (private 이면 0 으로 표시)                             |
| Archived = false      | `gh repo view OWNER/REPO --json isArchived`                        | archived 면 `eol` 배지 자동 부착                                               |
| Homepage URL          | `gh repo view OWNER/REPO --json homepageUrl`                       | `website_url` 후보. **단, 데모 버튼은 status 조건도 충족해야 노출** (아래 ⚠️) |
| Releases              | `gh api repos/OWNER/REPO/releases`                                 | **0 이면 `activity_status` = `developing` → 데모 버튼 안 뜸** ([github.ts:106](../src/lib/github.ts#L106)). ⚠️ 아래 참고 |
| LICENSE 파일          | `gh repo view OWNER/REPO --json licenseInfo`                       | 없으면 카드/상세 license 줄 빈칸                                               |
| `roadmap` 라벨 이슈   | `gh api 'repos/OWNER/REPO/issues?labels=roadmap&state=all'`        | 없으면 상세 페이지 Roadmap 블록 빈칸                                           |
| README `## Tagline`   | `gh api repos/OWNER/REPO/readme -H 'Accept: application/vnd.github.raw' \| grep '^##'` | 없으면 카드 한 줄 설명을 DB `tagline_*` 에 수동 입력해야 함                    |
| README `## Summary` 등 | 위와 동일                                                          | 없으면 상세 본문이 비어 보임 — DB `body_*` 직접 입력 또는 README 헤딩 추가     |
| 다국어                | README 안의 `Tagline-ko/ja/en`, `Summary-ko/ja/en` 헤딩 유무       | 미지원 언어는 `supported_locales` 에서 제외하면 카드 언어 칩에 안 뜸           |

> ⚠️ **"Try It" 데모 버튼이 뜨는 조건 (두 가지 동시 충족)** — [page.tsx:142-145](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L142):
> ```
> activity_status ∈ {poc, mvp, active}   AND   website_url 이 github.com 이 아닌 실제 URL
> ```
> `deriveActivityStatus` 는 **release 가 0개면 무조건 `developing`** ([github.ts:104-106](../src/lib/github.ts#L104)) 이라, **데모 사이트가 실제로 동작하더라도 release/status 없이는 버튼이 안 뜬다.** 데모 버튼을 노출하려면 다음 중 하나가 **필수**:
> - **(a)** semver 태그 release 발행 (`v0.1.0` → `active` 자동, [github.ts:119](../src/lib/github.ts#L119)) 또는 커스텀 태그 `poc`/`mvp`/`release` ([github.ts:112-115](../src/lib/github.ts#L112))
> - **(b)** DB `activity_status` 를 `poc`/`mvp`/`active` 로 직접 설정 — DB 값이 GitHub 자동판별보다 우선 ([page.tsx:99-102](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L99))
>
> (status 와 무관하게 GitHub 버튼은 public repo 면 항상 노출됨.)

> ⚠️ **상세 Overview(설명 본문) 블록이 뜨는 조건** — [page.tsx:178-179](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L178): DB `body_*` **또는** README `## Summary`/`## 프로젝트 개요` 중 **하나라도 있어야** 렌더. 둘 다 비면 상세 페이지에 배너 이미지 + 대시보드 통계만 나오고 **설명 텍스트가 전혀 없다.** → A-1 의 README Summary 섹션 추가가 "필수" 인 이유.

---

## A) 대상 레포 측 수정 (PR 형태로 별도 작업)

대상 레포는 별 git remote 이므로 그쪽 작업 디렉토리에서 commit. 권한이 없으면 owner 에게 PR 요청.

### A-1. README.md 에 자동 추출 섹션 추가 (필수)

추출 규칙 ([src/lib/github.ts:181](../src/lib/github.ts#L181)):

- `## Tagline` (또는 locale 접미사 `## Tagline-ko/-ja/-en`) — **카드 한 줄 설명**
- `## Summary` / `## 프로젝트 개요` / `## Overview` / `## 概要` / `## About` — **상세 페이지 본문**
- locale 접미사가 일반 이름보다 우선 매칭
- 다음 같은 레벨 헤딩 / `---` / `***` 까지가 한 섹션

README 최상단 H1 바로 아래에 다음 템플릿을 삽입:

```markdown
## Tagline-ko
(한국어 한 줄 설명 — 120자 이내 권장)

## Tagline-en
(English one-liner — under 120 chars)

## Tagline-ja
(日本語の一行説明)

## 프로젝트 개요
(상세 페이지 본문 — Markdown 가능. 무엇이 / 왜 / 어떻게)

## Summary-en
(English long-form description)

## Summary-ja
(日本語の詳細説明)

---
```

> 트레일링 `---` 가 추출 종료 마커. 기존 다른 섹션(`## 기술 스택` 등) 과 충돌 안 함.

### A-2. LICENSE 파일 추가 (권장)

상업적 활용/포크 정책 명확화 + 카드의 license 라인 채움. `gh repo edit --license` 가 아니라 **LICENSE 파일 자체를 commit** 해야 GitHub API `licenseInfo` 가 채워진다.

### A-3. Release 1개 발행 — **데모 버튼/상태 배지를 원하면 필수**

```bash
gh release create v0.1.0 --title "v0.1.0 — Initial PoC" --notes "MVP"
```

→ semver 태그면 `deriveActivityStatus` 가 `active` 로 자동 승격 ([src/lib/github.ts:119](../src/lib/github.ts#L119)). 커스텀 태그 `poc`/`mvp`/`release` 도 인식 ([src/lib/github.ts:112-115](../src/lib/github.ts#L112)).

**release 가 0개면 status 가 `developing` 으로 고정** ([src/lib/github.ts:106](../src/lib/github.ts#L106)) → 동작하는 데모가 있어도 "Try It" 버튼이 안 뜬다 (섹션 0 ⚠️ 참고). release 발행이 번거로우면 대안으로 **B-2 에서 DB `activity_status` 를 직접 `active` 로 설정** (DB 값이 자동판별보다 우선, [page.tsx:99-102](../src/app/[locale]/(main)/ai-services/[slug]/page.tsx#L99)).

### A-4. (선택) `roadmap` 라벨 + 향후 계획 이슈

```bash
gh label create roadmap --color FBCA04 --description "Workshop roadmap on aib.vote"
gh issue create --label roadmap --title "..." --body "..."
```

→ 상세 페이지 Roadmap 블록에 최신 5건 노출.

---

## B) knowai-space 측 등록

### B-1. 사전 충돌 확인 (선택, Supabase SQL editor)

```sql
select id, slug, is_active, display_order
from workshop_projects
where owner = '<OWNER>' and repo = '<REPO>';
```

행이 있으면 이미 등록된 것 — POST 는 409 반환. UPDATE 만 필요.

### B-2. Admin Panel 로 등록 (권장)

1. `https://www.aib.vote/admin` 접근 (계정의 `profiles.role` 이 `admin` 또는 `moderator` 여야 함).
2. Workshop 패널 ([src/app/admin/_components/workshop-panel.tsx](../src/app/admin/_components/workshop-panel.tsx)) → "새 프로젝트 추가" 폼.

필드 매핑 (스키마는 [src/lib/supabase/workshopTypes.ts:8-28](../src/lib/supabase/workshopTypes.ts#L8-L28)):

| 필드                                     | 입력 가이드                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `owner` (필수)                           | GitHub 계정/조직                                                                                       |
| `repo` (필수)                            | 레포 이름                                                                                              |
| `display_name`                           | 카드 제목. 비우면 `repo` 사용                                                                          |
| `tagline_ko` / `tagline_ja` / `tagline_en` | A-1 패치 머지 후 비워두면 README 자동 채움. 머지 전이면 직접 입력                                       |
| `body_ko` / `body_ja` / `body_en`        | 동일                                                                                                   |
| `image_url`                              | OG 이미지 또는 R2 (`cdn.aib.vote/...`) 업로드 URL. 없으면 비움 (이미지 없는 카드)                       |
| `website_url`                            | 데모/홈페이지 URL                                                                                      |
| `supported_locales`                      | `["ko","ja","en"]` 부분집합. README 에 해당 locale 섹션이 있는 것만                                      |
| `activity_status`                        | 비움 (NULL — GitHub releases 로 자동 판별). 강제 override 가 필요할 때만 `developing/poc/mvp/active/eol` |
| `display_order`                          | 비움 (현재 최댓값 + 1 자동)                                                                            |
| `is_active`                              | POST 핸들러가 항상 `true` 로 INSERT                                                                    |

### B-3. CLI 로 등록 (B-2 대신, 자동화/스크립트용)

관리자 세션 쿠키 또는 service role 환경에서:

```bash
curl -X POST https://www.aib.vote/api/admin/workshop \
  -H 'Content-Type: application/json' \
  -b 'sb-...-auth-token=...' \
  -d '{
    "owner": "<OWNER>",
    "repo": "<REPO>",
    "display_name": "<title>",
    "website_url": "<demo url>",
    "supported_locales": ["ko","en"]
  }'
```

응답 `201 { item: { ..., slug } }`. URL 은 `https://www.aib.vote/ko/ai-services/<slug>`.

### B-4. `/ai-services` 캐시 무효화 주의

POST 핸들러는 [src/app/api/admin/workshop/route.ts:95](../src/app/api/admin/workshop/route.ts#L95) 에서 `revalidatePath('/workshop')` 만 호출 — 실제 노출 경로는 `/ai-services` 라 신규 카드가 **즉시 안 보일 수 있음** (ISR 24h, [page.tsx:23](../src/app/[locale]/(main)/ai-services/page.tsx#L23)).

해소 방법 중 택1:

- (a) Vercel 대시보드 → Deployments → 해당 path 수동 purge
- (b) 24h 자연 만료 대기
- (c) 핸들러에 `revalidatePath('/ai-services')` 추가하는 별 PR — 가장 안전한 영구 해결

---

## C) 검증 절차 (등록 후)

1. **DB 확인**:
    ```sql
    select slug, display_name, supported_locales, activity_status, is_active
    from workshop_projects where slug = '<slug>';
    ```
2. **GitHub fetch 확인**: 로컬 (`npm run dev`, `GITHUB_TOKEN` 설정 — [src/lib/github.ts:97](../src/lib/github.ts#L97)) 에서 `http://localhost:3000/ko/ai-services` 진입 → 카드의 stars/contributors/issues 가 0 이 아닌 실수치인지.
3. **Tagline 자동 채움**: 카드 한 줄 설명이 README A-1 tagline 으로 채워졌는지 (DB tagline 이 비어있어도).
4. **상세 페이지** `/ko/ai-services/<slug>`:
    - 본문 = README `## 프로젝트 개요` 섹션
    - Roadmap = A-4 적용 시 이슈 목록, 아니면 비어있는 안내
    - "Try It" 데모 버튼 → `website_url` (단 `activity_status` 가 poc/mvp/active 일 때만 노출 — 섹션 0 ⚠️). 안 보이면 status 부터 확인.
    - License = A-2 적용 시 표시
5. **다국어 fallback**: `/en/ai-services/<slug>` → `Tagline-en` / `Summary-en` 노출 확인. 누락 시 ko 가 fallback ([src/lib/supabase/workshopTypes.ts:32-42](../src/lib/supabase/workshopTypes.ts#L32-L42)).
6. **운영 반영**: B-4 절차 중 택1.

---

## D) 권장 작업 순서

1. **대상 레포 측 PR**: A-1 (README Tagline/Summary 섹션) — 머지 즉시 카드 자동 채움. A-2 LICENSE, A-3 release, A-4 roadmap 라벨은 점진 가능.
2. **knowai-space 측 등록**: B-2 Admin UI (또는 B-3 curl).
3. **즉시 노출 필요 시**: B-4(c) 의 한 줄 추가 PR.
4. **검증**: C 절차.

---

## E) 예시 — `Jaemnani/bible-search` (실제 케이스)

위 절차를 그대로 적용한 사례. 새 레포 등록 시 OWNER/REPO 부분만 치환하면 됨.

### 점검 결과 (섹션 0)

| 항목                  | 현재                                            | 조치                                                              |
| --------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| Visibility            | ✅ Public                                       | -                                                                 |
| Archived              | ✅ false                                        | -                                                                 |
| Homepage URL          | `https://bible-search-nine.vercel.app`          | `website_url` 로 사용 (동작하는 데모 있음)                        |
| Releases              | ❌ 0개                                          | **A-3 또는 activity_status 설정 필수** — 안 하면 데모 버튼 안 뜸  |
| LICENSE               | ❌ 없음                                         | A-2 적용 권장                                                     |
| `roadmap` 라벨        | ❌ 없음                                         | A-4 적용 (선택)                                                   |
| README `## Tagline`   | ❌ 없음                                         | A-1 패치 필수                                                     |
| README `## Summary`   | ❌ 없음 (`기술 스택`, `검색 파이프라인` 만 존재) | A-1 패치 필수                                                     |
| 다국어                | ko 만                                           | `supported_locales = ["ko","en"]` (영어는 README 패치로 추가)      |

### A-1 README 패치 (bible-search 전용 내용)

`# 📖 Bible Search` 바로 아래 삽입:

```markdown
## Tagline-ko
감정·주제 기반 성경 구절 시맨틱 검색. "외로워요", "두려워요" 같은 일상적 감정으로 관련 성경 말씀을 추천받습니다.

## Tagline-en
Emotion-aware semantic search for Bible verses. Find scripture by everyday feelings like "I feel lonely" or "I'm afraid."

## Tagline-ja
感情・テーマに基づく聖書節セマンティック検索。「寂しい」「怖い」など日常の感情からみことばを提案します。

## 프로젝트 개요
BM25(희소) + Gemini 임베딩(밀집) 하이브리드 검색으로 성경 30,944개 구절에서 사용자 감정/주제에 어울리는 본문을 찾아 Gemini 2.5 Flash Lite 로 리랭킹합니다. 정적 벡터 파일 + Vercel 서버리스로 호스팅 비용 0 에 가깝게 운영합니다.

## Summary-en
Hybrid BM25 + Gemini embeddings search across 30,944 verses, reranked by Gemini 2.5 Flash Lite. Static vector files on Vercel keep hosting cost near zero.

## Summary-ja
30,944節を対象に BM25 と Gemini エンベディングのハイブリッド検索を行い、Gemini 2.5 Flash Lite で再ランキング。静的ベクトルファイル + Vercel でホスティングコストをほぼゼロに抑えています。

---
```

### B-2 등록 폼 입력값

| 필드                | 값                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------ |
| `owner`             | `Jaemnani`                                                                           |
| `repo`              | `bible-search`                                                                       |
| `display_name`      | `Bible Search`                                                                       |
| `tagline_*`         | 비움 (A-1 머지 후 자동). 머지 전 등록 시 위 ko/en/ja 한 줄을 복사 입력                |
| `body_*`            | 동일                                                                                 |
| `image_url`         | 비움 또는 R2 업로드 후 입력                                                          |
| `website_url`       | `https://bible-search-nine.vercel.app`                                               |
| `supported_locales` | `["ko","en"]` (JA 임베딩 없음 — A-1 의 JA tagline 은 fallback 표시용)                  |
| `activity_status`   | **`active` 또는 `poc` 직접 입력 권장** — bible-search 는 release 0개라 비워두면 `developing` 으로 고정돼 "Try It" 데모 버튼이 안 뜬다. (A-3 release 를 발행하면 비워둬도 됨) |

### B-3 등가 curl

```bash
curl -X POST https://www.aib.vote/api/admin/workshop \
  -H 'Content-Type: application/json' \
  -b 'sb-...-auth-token=...' \
  -d '{
    "owner": "Jaemnani",
    "repo": "bible-search",
    "display_name": "Bible Search",
    "website_url": "https://bible-search-nine.vercel.app",
    "supported_locales": ["ko","en"]
  }'
```

등록 후 URL: `https://www.aib.vote/ko/ai-services/bible-search`.
