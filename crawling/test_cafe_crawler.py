"""
카페 크롤러 오프라인 테스트 — 네이버 접속 없이 가짜 드라이버로 목록 API·본문 찾기·이력 보존을 검증
  python -m pytest crawling -q   (selenium, numpy, pytest 필요)
"""
import importlib.util
import os
import json
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("pyautogui", "webdriver_manager", "webdriver_manager.chrome"):
    m = types.ModuleType(name)
    m.ChromeDriverManager = object
    sys.modules.setdefault(name, m)

spec = importlib.util.spec_from_file_location("crawler", f"{ROOT}/crawling/01_dh_caffe_crawling_with_auto_login.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class El:
    def __init__(self, text):
        self.text = text


class FakeDriver:
    """url → 페이지 {'body': str, 'iframe': bool, 'container': str|None}"""

    def __init__(self, pages):
        self.pages, self.url, self.in_frame, self.visited = pages, None, False, []
        self.switch_to = types.SimpleNamespace(default_content=self._default, frame=self._frame)

    def _default(self):
        self.in_frame = False

    def _frame(self, _):
        self.in_frame = True

    def get(self, url):
        self.url, self.in_frame = url, False
        self.visited.append(url)

    def _page(self):
        return self.pages.get(self.url, {})

    def find_element(self, by, value):
        if value == "body":
            return El(self._page().get("body", ""))
        raise c.TimeoutException("no element")

    def find_elements(self, by, value):
        p = self._page()
        if value == "cafe_main":
            return [object()] if p.get("iframe") else []
        if value in c.CONTAINER_CLASSES:
            # iframe 페이지는 프레임 안에서만 본문이 보인다
            if p.get("container") and bool(p.get("iframe")) == self.in_frame and p.get("cls", "se-main-container") == value:
                return [El(p["container"])]
        return []


def list_url(page):
    return c.LIST_API.format(cafe=c.CAFE_ID, menu=c.MENU_ID, page=page)


def art_url(i, aid):
    return c.ARTICLE_URLS[i].format(cafe=c.CAFE_ID, menu=c.MENU_ID, aid=aid)


def api(articles, has_next):
    return {"body": json.dumps({"message": {"result": {"articleList": articles, "hasNext": has_next}}})}


def article_text(bs, reh, win):
    return "\n".join([
        "동행복권 로또6/45", "1. 볼세트", str(bs),
        "2. 모의추첨번호(리허설)", " ".join(map(str, reh)),
        "3. 당첨번호(추첨순)", " ".join(map(str, win)),
        "4. 당첨번호(오름차순)", " ".join(map(str, sorted(win[:6]))),
    ])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda *_: None)
    monkeypatch.setattr(c, "_fetch_official_details", lambda r: np.array([str(r)] * len(c.DETAIL_COLUMNS)))


def test_list_articles_pages_dedup_and_filter():
    d = FakeDriver({
        list_url(1): api([
            {"articleId": 300, "subject": "동행복권 로또6/45 제1244회 추첨 리허설 정보 (2026.10.03)"},
            {"articleId": 299, "subject": "[공지] 커뮤니티 규정"},
            {"articleId": 298, "subject": "동행복권 로또6/45 제 1243 회 추첨 리허설 정보"},
        ], True),
        list_url(2): api([
            {"articleId": 150, "subject": "동행복권 로또6/45 제1243회 추첨 리허설 정보 (중복 옛 글)"},
            {"articleId": 149, "subject": "동행복권 로또6/45 제1242회 추첨 리허설 정보"},
        ], False),
    })
    out = c.list_articles(d)
    assert out.tolist() == [["300", "1244"], ["298", "1243"], ["149", "1242"]]
    assert d.visited == [list_url(1), list_url(2)]


def test_open_article_new_ui_old_iframe_and_timeout():
    d = FakeDriver({art_url(0, "7"): {"container": "새UI 본문"}})
    assert c.open_article(d, "7").text == "새UI 본문"

    d = FakeDriver({art_url(0, "8"): {}, art_url(1, "8"): {"container": "옛UI 본문", "iframe": True, "cls": "ContentRenderer"}})
    assert c.open_article(d, "8", timeout=0.05).text == "옛UI 본문"

    d = FakeDriver({})
    with pytest.raises(c.TimeoutException):
        c.open_article(d, "9", timeout=0.05)


def test_merge_history_keeps_precafe_rounds_and_order():
    hist = np.array([
        ["", "733", "d"], ["", "734", "d"],
        ["3", "835", "reh"], ["3", "835", "real"],
    ])
    crawled = [np.array(["3", "835", "reh"]), np.array(["3", "835", "real"]),
               np.array(["1", "836", "reh"]), np.array(["1", "836", "real"])]
    out = [list(r) for r in c.merge_history(crawled, hist)]
    assert [r[1] for r in out] == ["733", "734", "835", "835", "836", "836"]
    assert [r[2] for r in out if r[1] == "836"] == ["reh", "real"]


def test_crawl_cafe_end_to_end():
    width = 10 + len(c.DETAIL_COLUMNS)
    def row(bs, rnd, nums):
        return np.array([bs, str(rnd), c.round_to_date(rnd), *map(str, nums)] + [""] * (width - 10))
    history = np.array([
        row("", 733, [1, 2, 3, 4, 5, 6, 7]),                       # 카페 개설 전 (단일 행)
        row("2", 835, [1, 2, 3, 4, 5, 6, 7]), row("2", 835, [8, 9, 10, 11, 12, 13, 14]),
    ])
    d = FakeDriver({
        list_url(1): api([
            {"articleId": 903, "subject": "동행복권 로또6/45 제837회 추첨 리허설 정보"},
            {"articleId": 902, "subject": "동행복권 로또6/45 제836회 추첨 리허설 정보"},
            {"articleId": 901, "subject": "동행복권 로또6/45 제835회 추첨 리허설 정보"},
        ], False),
        art_url(0, "902"): {"container": article_text(4, [1, 9, 17, 25, 33, 41, 45], [2, 10, 18, 26, 34, 42, 44])},
        art_url(0, "903"): {"container": "형식이 다른 글"},
    })
    out = c.crawl_cafe(d, history)
    rounds = [int(r[1]) for r in out]
    assert rounds == [733, 835, 835, 836, 836]          # 837 은 형식 오류로 건너뜀, 733 보존
    r836 = [r for r in out if r[1] == "836"]
    assert list(r836[0][:10]) == ["4", "836", c.round_to_date(836), "1", "9", "17", "25", "33", "41", "45"]
    assert list(r836[1][3:10]) == ["2", "10", "18", "26", "34", "42", "44"]
    assert all(len(r) == width for r in out)
    assert not any(art_url(0, "901") == u for u in d.visited)   # 이미 있는 835 는 다시 안 읽음


def test_no_hardcoded_secrets():
    src = open(f"{ROOT}/crawling/01_dh_caffe_crawling_with_auto_login.py").read()
    assert 'os.getenv("NAVER_PW", "")' in src and 'os.getenv("NAVER_ID", "")' in src
    assert "AIza" not in src


def test_real_csv_roundtrip_is_lossless():
    """레포 CSV 의 카페 회차가 전부 목록에 있을 때, 저장 결과가 원본과 같아야 한다 (733~834회 보존)."""
    raw = np.loadtxt(c.HISTORY_PATH, delimiter=",", dtype=str)
    rounds = sorted({int(r) for r in raw[:, 1] if int(r) >= 835}, reverse=True)
    arts = [{"articleId": 10000 + r, "subject": f"동행복권 로또6/45 제{r}회 추첨 리허설 정보"} for r in rounds]
    pages = {list_url(i + 1): api(arts[i * 50:(i + 1) * 50], (i + 1) * 50 < len(arts))
             for i in range((len(arts) + 49) // 50)}
    out = np.array(c.crawl_cafe(FakeDriver(pages), raw))
    assert np.array_equal(out, raw)
