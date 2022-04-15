import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":

    header = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/100.0.4896.75 Safari/537.36"}

    urlList = []

    i = 0

    print("-----Start of Crawl-----")

    while True:
        rootUrl = "https://phishtank.com/phish_search.php?page={page}&active=y&verified=u"
        content = requests.get(rootUrl.format(page=i), headers=header).text

        rootSoup = BeautifulSoup(content, "html.parser")
        rows = rootSoup.findAll("tr")[1:]

        if len(rows) <= 2:
            break

        print("page={page}".format(page=i))

        for row in rows:
            cols = row.find_all("td", class_="value")
            col = cols[1]
            url = col.get_text().split("added")[0]
            urlList.append(url)

        i += 1

    file = open("urls.txt", "a")

    for url in urlList:
        file.write(url + "\n")

    file.close()

    print("-----End of Crawl-----")
