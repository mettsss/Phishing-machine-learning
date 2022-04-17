import requests
import codecs
import pandas as pd
from bs4 import BeautifulSoup


# read raw data and convert to dict type
def load_raw_dataset(filename=None):
    f = codecs.open(filename, 'r', 'utf-8', errors='replace')
    lines = f.readlines()
    f.close()

    raw_dataset = []

    for line in lines:
        strs = line.strip().split('\t')
        label = strs[0]
        url = strs[1]
        data = {'url': url, 'label': label}
        raw_dataset.append(data)

    return raw_dataset


def data_crawl(url=None, data=None):
    try:
        r = requests.get(url, headers=header, timeout=2.0)
        content = r.text
        soup = BeautifulSoup(content, 'html.parser')

        # feature: cookie_len
        cookie = r.cookies.get_dict()
        data['cookie_len'] = len(cookie)

        # feature: form_num
        form_num = len(soup.findAll('form'))
        data['form_num'] = form_num

        # feature: anchor_num
        anchor_num = len(soup.findAll('a'))
        data['anchor_num'] = anchor_num

        # feature: input_email
        input_email = soup.findAll('input', type="email")
        if len(input_email) > 0:
            data['input_email'] = str(True)
        else:
            data['input_email'] = str(False)

        # feature: input_password
        input_password = soup.findAll('input', type="password")
        if len(input_password) > 0:
            data['input_password'] = True
        else:
            data['input_password'] = False

        # feature: hidden
        hidden = soup.findAll('input', type="hidden")
        if len(hidden) > 0:
            data['hidden'] = True
        else:
            data['hidden'] = False

    # url is not reachable
    except Exception:
        pass


if __name__ == "__main__":
    header = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/100.0.4896.75 Safari/537.36"}

    train_file = '../dataset/small_dataset/train.txt'
    test_file = '../dataset/small_dataset/test.txt'

    train_dataset = load_raw_dataset(filename=train_file)
    test_dataset = load_raw_dataset(filename=test_file)

    i = 0
    j = 0

    for train_data in train_dataset:
        train_url = train_data['url']
        data_crawl(url=train_url, data=train_data)
        print('train {num1}, {num2} left'.format(num1=i, num2=len(train_dataset) - i))
        i += 1

    train_df = pd.DataFrame(train_dataset)
    train_output_path = 'train.csv'
    train_df.to_csv(train_output_path, sep=",", index=False, header=True)

    for test_data in test_dataset[0:10000]:
        test_url = test_data['url']
        data_crawl(url=test_url, data=test_data)
        print('test {num1}, {num2} left'.format(num1=j, num2=len(test_dataset) - j))
        j += 1

    test_df = pd.DataFrame(test_dataset)
    test_output_path = 'test.csv'
    test_df.to_csv(test_output_path, sep=",", index=False, header=True)
