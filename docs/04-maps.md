# 04 · 지도 (타일 + 외부 지도 딥링크)

지도 표시와 "외부 지도 앱으로 열기"를 담당하는 연동. 이 프로젝트는 **API 키 없이** 동작한다:
오픈 타일 + 클라이언트 렌더링 + keyless 딥링크.

## 4-1. 지도 렌더링 — MapLibre GL + OpenFreeMap

| 항목 | 값 |
|---|---|
| 렌더러 | `maplibre-gl` (오픈소스, Mapbox GL JS 포크) |
| 타일/스타일 | `https://tiles.openfreemap.org/styles/liberty` (무료, 키없음) |
| 컨트롤 | `NavigationControl`, `ScaleControl`, 마커/팝업 |

패턴: 클라이언트 컴포넌트에서 `new maplibregl.Map({ style, center, zoom })` 생성, 좌표 배열로
`Marker`/`Popup` 추가. 키가 없으므로 env 설정 불필요.

> OpenFreeMap은 무료·무키지만 best-effort SLA. 트래픽 큰 프로덕션은 자체 타일 호스팅 또는
> 유료 타일(Maptiler/Mapbox 등)로 교체 가능 — `style` URL만 바꾸면 됨.

코드 위치(예): `web/src/components/property-map.tsx`, `web/src/components/property-location.tsx`.

## 4-2. 외부 지도 딥링크 (네이버/카카오/구글)

"네이버지도/카카오맵/구글맵에서 보기" 링크. **API가 아니라 URL 스킴**이라 키 불필요.

| 앱 | 주소 검색(핀 잘 잡힘) | 좌표 폴백 |
|---|---|---|
| 네이버 | `https://map.naver.com/p/search/{주소}` | `.../p/search/{lat},{lng}` |
| 카카오 | `https://map.kakao.com/?q={주소}` | `https://map.kakao.com/link/map/{label},{lat},{lng}` |
| 구글 | `https://www.google.com/maps/search/?api=1&query={주소}` | `https://www.google.com/maps?q={lat},{lng}` |

원칙:
- **주소 텍스트가 있으면 주소 검색**(건물명까지 있으면 정확히 핀), 없으면 좌표 폴백.
- `encodeURIComponent`로 쿼리 인코딩. `target="_blank" rel="noopener noreferrer"`.

> **주의:** 정식 Maps API(정적지도/길찾기/지오코딩)를 쓰려면 그건 키가 필요한 **별도 연동**이다.
> 이 프로젝트의 `NEXT_PUBLIC_NAVER_MAP_*` 키는 **현재 코드에서 미사용**(딥링크만 사용) — 죽은 값이므로
> 실사용 시 제거하거나, 정식 API 도입 시 [03-enrichment](03-enrichment-apis.md) 패턴으로 옮길 것.

## 재사용 체크리스트
- [ ] 타일 스타일 URL 선택(무료 OpenFreeMap / 유료 키 기반).
- [ ] 좌표 + (가능하면)주소 텍스트 확보 → 딥링크 정확도 ↑.
- [ ] 딥링크는 키없이 URL만. 정식 Maps API 도입 시에만 키 관리.
