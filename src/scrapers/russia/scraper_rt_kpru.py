import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


class RT:
    def __init__(self):
        self.website = "kpru"
        if self.website == "kpru":
            self.base_url = "https://www.kp.ru/"
        elif self.website == "rt":
            self.base_url = "https://russian.rt.com/"

        self.semaphore = asyncio.Semaphore(5)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "Referer": self.base_url,
            "Origin": self.base_url,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
        }
        

    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                return await response.json()  # directly get JSON
    
    async def fetch2(self, session, url):
        async with self.semaphore:
            async with session.get(url, headers = self.headers) as response:
                return await response.text()


    async def extract_data(self):
        all_urls = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for i in range(10):
                page_number = 711 - i
                url = f"https://s02.api.yc.kpcdn.net/content/api/1/pages/get.json?pages.direction=page&pages.number={page_number}&pages.spot=0&pages.target.class=111&pages.target.id=0"
                data = await self.fetch(session, url)
                top_childs = data.get("childs", [])

                def get_article_urls(blocks):
                    urls = []
                    for block in blocks:
                        if "@type" in block and "article" in block["@type"] and "issue" in block and "@id" in block:
                            urls.append(f"https://www.kp.ru/daily/{block['issue']}/{block['@id']}/")
                        if "childs" in block:
                            urls.extend(get_article_urls(block["childs"]))
                    return urls

                urls = get_article_urls(top_childs)
                urls = [urljoin(self.base_url, u) for u in urls]
                all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls})


    async def extract_text(self, session):
        self.df_titles = pd.read_csv("../../data/russia/kpru0.csv")
        tasks = [self.fetch2(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        all_dates = []
        all_titles = []
        for soup in soups:
            titles = soup.find("h1", attrs = {"class": "sc-j7em19-3 eyeguj"}).text.strip()
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            # dates = [d.find("time", attrs = {"class": "date"}).text.strip() for d in soup.find_all("div", attrs = {"class": "article__date"})]
            # dates = soup.find("h1", attrs = {"span": "sc-j7em19-1 dtkLMY"}).text.strip()
            all_texts.append(" ".join(paragraphs))
            # all_dates.append(" ".join(dates))
            all_titles.append(" ".join(titles))

        self.df_text = pd.DataFrame({"title": all_titles, "text": all_texts})
        self.df_text.to_csv("../../../data/russia/kprunew.csv")
    

    async def process_website(self):
        await self.extract_data()
        async with aiohttp.ClientSession(headers=self.headers) as session:
            await self.extract_text(session)



async def main():
    scraper = RT()
    await scraper.process_website()


if __name__ == "__main__":
    asyncio.run(main())
