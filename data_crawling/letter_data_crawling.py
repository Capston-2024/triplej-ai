import requests
from bs4 import BeautifulSoup
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

GSPREAD_KEY = os.getenv('GSPREAD_KEY')
BASE_URL = os.getenv('BASE_URL')
LIST_URL = f"{BASE_URL}/essay/home"

# --- google spreadsheet auth config ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GSPREAD_KEY, scope)
client = gspread.authorize(creds)

# open google spreadsheet
spreadsheet = client.open('letters') 
worksheet = spreadsheet.sheet1

# set headers
worksheet.update([["url", "works", "letter-kr"]], "A1")

# --- data crawling ---
response = requests.get(LIST_URL)
soup = BeautifulSoup(response.text, "html.parser")

# extract letter urls
links = []
for a_tag in soup.find_all("a", href=True):
    href = a_tag["href"]
    if href.startswith("/blog/"):
        full_url = BASE_URL + href
        links.append(full_url)

# extract letter contents
row_data = []
for url in links:
    res = requests.get(url)
    res.encoding = 'utf-8'
    detail_soup = BeautifulSoup(res.text, "html.parser")

    work_div = detail_soup.select_one("div.page-wrapper div.section.pd-top-80px.pd-bottom-0px.blog div.container-default.w-container div.w-layout-blockcontainer.container-90.w-container div.div-block-514 div.div-block-499 div.text-block-396")
    work = work_div.get_text(strip=True) if work_div else ""

    letter_div = detail_soup.select_one("div.page-wrapper div.section.small.blog._2 div.container-default.w-container div.responsive-container._700px-tablet div.w-layout-grid.grid-2-columns.blog-post-grid.blog div.w-tabs div.w-tab-content div.w-tab-pane.w--tab-active div.inner-container._664px._100---tablet div.rich-text.blog.no-copy.w-richtext")
    if letter_div:
        for strong_tag in letter_div.find_all("strong"):
            strong_tag.decompose()
    
        letter = letter_div.get_text(strip=True)
    else:
        letter = ""
    
    row_data.append([url, work, letter])  

# upload data
worksheet.update(row_data, f"A2")
