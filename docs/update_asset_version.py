# -*- coding: utf-8 -*-
"""更新 docs/index.html 中静态资源的版本号，避免浏览器缓存旧脚本。

网页版最容易踩的坑：改了 dither.js 之后，index.html 是新的、JS 还是缓存的旧版，
于是出现"选项是新的但处理失败"这类怪现象。给资源加内容哈希当版本号就能根治。

每次改动 docs/ 下的 .js / .css 之后运行一次：

    python docs/update_asset_version.py

它会把 index.html 里所有 `xxx.js?v=...` / `xxx.css?v=...` 统一换成当前内容哈希。
内容没变则版本号不变（幂等）；新增引用只要先写成 `xxx.js?v=1` 就会被接管。
"""
import hashlib
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ASSETS = ["delaunay.js", "dither.js", "textpixel.js", "app.js", "style.css"]
PATTERN = re.compile(r'((?:src|href)="[^"]+?\.(?:js|css))\?v=[^"]*"')


def content_version() -> str:
    h = hashlib.md5()
    for name in ASSETS:
        h.update((DOCS / name).read_bytes())
    return h.hexdigest()[:8]


def main() -> int:
    ver = content_version()
    idx = DOCS / "index.html"
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
