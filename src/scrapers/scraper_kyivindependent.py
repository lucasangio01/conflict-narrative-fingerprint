import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


class KyivPost:
    def __init__(self):
        self.api_url = "https://kyivindependent.com/ghost/api/content/posts/"
        self.params = {
            "key": "f19c122e3d36483beb01f3013a",
            "limit": 8,
            "include": "tags,authors_count,authors,claps,comments_count",
            "filter": "tag:opinion",
            "formats": "html",
            "order": "published_at DESC"
        }
        self.semaphore = asyncio.Semaphore(5)



    async def fetch(self, session, page):
        self.params["page"] = page
        async with session.get(self.api_url, params = self.params) as response:
            response.raise_for_status()
            return await response.json()


    async def fetch_html(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self):
        all_titles = []
        all_urls = []
        page = 1
        async with aiohttp.ClientSession() as session:
            for page in range(1, 11): # 11 pagine
                data = await self.fetch(session, page)
                posts = data.get("posts", [])
                if not posts:
                    break
                for post in posts:
                    all_titles.append(post["title"])
                    all_urls.append(f'https://kyivindependent.com/{post["slug"]}/')
        self.df_titles = pd.DataFrame({"url": all_urls, "title": all_titles})


    async def extract_text(self, session):
        titles = self.df_titles["title"]
        tasks = [self.fetch_html(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        for soup in soups:
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            all_texts.append(" ".join(paragraphs))
        self.df_text = pd.DataFrame({"title": titles, "text": all_texts})
        self.df_text.to_csv("../../data/ukraine/kyiv_independent.csv")


    async def process_website(self, session):
        await self.extract_data()
        await self.extract_text(session)



async def main():
    scraper = KyivPost()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
