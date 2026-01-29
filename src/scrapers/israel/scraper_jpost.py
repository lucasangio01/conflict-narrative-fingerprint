import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os


class JPost:
    def __init__(self):
        self.base_url = "https://www.jpost.com/opinion"
        self.semaphore = asyncio.Semaphore(3)
        self.csv_path = "../../../data/israel/1_original/jpost.csv"
        if os.path.exists(self.csv_path):
            self.df_text = pd.read_csv(self.csv_path)
            if "Unnamed: 0" in self.df_text.columns:
                self.df_text = self.df_text.drop(columns=["Unnamed: 0"])
            self.existing_titles = set(self.df_text["title"])
        else:
            self.df_text = pd.DataFrame(columns=["title", "date", "text"])
            self.existing_titles = set()


    async def fetch(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()


    async def extract_data(self, session):
        all_urls = []
        for i in range(1, 4):
            url = f"https://www.jpost.com/opinion?page={i}"
            html = await self.fetch(session, url)
            soup = await asyncio.to_thread(BeautifulSoup, html, "html.parser")
            urls = [urljoin(self.base_url, t["href"]) for t in soup.find_all("a", attrs = {"class": "article-item-content-link"}) if t is not None]
            all_urls.extend(urls)
        self.df_titles = pd.DataFrame({"url": all_urls})


    async def extract_text(self, session):
        tasks = [self.fetch(session, row.url) for row in self.df_titles.itertuples()]
        html_pages = await asyncio.gather(*tasks)
        parse_tasks = [asyncio.to_thread(BeautifulSoup, html, "html.parser") for html in html_pages]
        soups = await asyncio.gather(*parse_tasks)
        all_texts = []
        all_titles = []
        all_dates = []
        for soup in soups:
            title_list = [t.text.strip() for t in soup.find_all("h1", attrs={"class": "article-main-title"})]
            title = " ".join(title_list)
            paragraphs = [p.text.strip() for p in soup.find_all("p")]
            date_list = [d.text.strip() for d in soup.find_all("time")][0]
            date = " ".join(date_list)

            if title not in self.existing_titles and title != "":
                all_titles.append(title)
                all_texts.append(" ".join(paragraphs))
                all_dates.append(date)
                self.existing_titles.add(title)

        if all_titles:
            new_df = pd.DataFrame({"title": all_titles, "date": all_dates, "text": all_texts})
            self.df_text = pd.concat([self.df_text, new_df], ignore_index=True).sort_values(by="date", ascending=False)
            self.df_text.drop_duplicates(subset="title", inplace=True)
            self.df_text.to_csv(self.csv_path, index=False)
    
    async def process_website(self, session):
        await self.extract_data(session)
        await self.extract_text(session)



async def main():
    scraper = JPost()
    async with aiohttp.ClientSession() as session:
        await scraper.process_website(session)


if __name__ == "__main__":
    asyncio.run(main())
