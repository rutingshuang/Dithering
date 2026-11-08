# -*- coding: utf-8 -*-
"""更新 docs/index.html 中静态资源的版本号，避免浏览器缓存旧脚本。

网页版最容易踩的坑：改了 dither.js 之后，index.html 是新的、JS 还是缓存的旧版，
于是出现"选项是新的但处理失败"这类怪现象。给资源加内容哈希当版本号就能根治。

每次改动 docs/ 下的 .js / .css 之后运行一次：

    python docs/update_asset_version.py

它会把 index.html 里所有 `xxx.js?v=...` / `xxx.css?v=...` 统一换成当前内容哈希。
内容没变则版本号不变（幂等）；新增引用只要先写成 `xxx.js?v=1` 就会被接管。

改了 **index.html 的结构**（加/删/挪控件）时，还要递增外壳版本：

    python docs/update_asset_version.py --bump

`?v=` 管不了 HTML 自己——它就是入口，没法给自己挂哈希。所以 index.html 里内联
一个 `window.__DITHER_SHELL__`，app.js 里对应 `const SHELL_BUILD`；两者不等时
app.js 会带时间戳 query 强拉一次新外壳。--bump 负责把这两处同时加一，别手动改。
"""
import hashlib
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ASSETS = ["delaunay.js", "dither.js", "textpixel.js", "app.js", "style.css"]
PATTERN = re.compile(r'((?:src|href)="[^"]+?\.(?:js|css))\?v=[^"]*"')
SHELL_HTML = re.compile(r"window\.__DITHER_SHELL__\s*=\s*'([^']*)'")
SHELL_JS = re.compile(r"const SHELL_BUILD\s*=\s*'([^']*)'")


def content_version() -> str:
    h = hashlib.md5()
    for name in ASSETS:
        h.update((DOCS / name).read_bytes())
    return h.hexdigest()[:8]


def bump_shell(idx: pathlib.Path, app: pathlib.Path) -> str:
    """把 index.html 与 app.js 里的外壳版本同时加一，返回新版本号。"""
    html = idx.read_text(encoding="utf-8")
    m = SHELL_HTML.search(html)
    if not m:
        print("index.html 里没找到 window.__DITHER_SHELL__，先按模板补上")
        return ""
    try:
        nxt = str(int(m.group(1)) + 1)
    except ValueError:
        print(f"__DITHER_SHELL__ 不是整数：{m.group(1)!r}")
        return ""

    js = app.read_text(encoding="utf-8")
    if not SHELL_JS.search(js):
        print("app.js 里没找到 const SHELL_BUILD，先按模板补上")
        return ""

    idx.write_text(
        SHELL_HTML.sub(lambda _: f"window.__DITHER_SHELL__ = '{nxt}'", html, count=1),
        encoding="utf-8",
    )
    app.write_text(
        SHELL_JS.sub(lambda _: f"const SHELL_BUILD = '{nxt}'", js, count=1),
        encoding="utf-8",
    )
    return nxt


def main() -> int:
    idx = DOCS / "index.html"
    if "--bump" in sys.argv[1:]:
        nxt = bump_shell(idx, DOCS / "app.js")
        if not nxt:
            return 1
        print(f"外壳版本 -> {nxt}")

    ver = content_version()
    html = idx.read_text(encoding="utf-8")
    new, n = re.subn(PATTERN, lambda m: f'{m.group(1)}?v={ver}"', html)
    if n == 0:
        print("没有找到带 ?v= 的资源引用；请先给引用手写一次（如 dither.js?v=1）")
        return 1
    if new == html:
        print(f"版本号已是最新：{ver}（共 {n} 处引用）")
        return 0
    idx.write_text(new, encoding="utf-8")
    print(f"已更新 {n} 处资源版本号 -> {ver}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
