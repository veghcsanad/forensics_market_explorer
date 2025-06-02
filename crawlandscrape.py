# This script crawls and scrapes darknet market pages for cannabis product listings.

import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
import re

# Set up a random user agent and Tor proxy for anonymity
ua = UserAgent()
proxy = {
    "https": "socks5h://127.0.0.1:9050",
    "http": "socks5h://127.0.0.1:9050"
}

def crawl_page(url, cookie):
    """
    Sends a GET request to the given URL using Tor proxy and custom headers.
    Returns the HTTP response.
    """
    header = {
        "User-Agent": ua.random,
        "Referer": url
    }
    response = requests.get(url, proxies=proxy, cookies=cookie, headers=header)
    return response

def scrape_products(content):
    """
    Parses the HTML content to extract product information.
    Returns a list of unique product dictionaries.
    """
    soup = BeautifulSoup(content, 'html.parser')
    unique_products = set()
    products = []
    # Find the main product listings container
    products_div = soup.find('div', {"class": "Messbar"})
    all_products = products_div.find_all('div', class_='startw')

    for product in all_products:
        # Extract product title
        title = product.contents[2].strip()
        # Extract seller name
        name = product.find('span', style="color: #F74A3F").text.strip()
        # Extract price
        price = product.find('span', style="color: #B74A3F").text.strip()
        # Extract origin (destination)
        origin_div = soup.find('div', class_=['product-sale-Escrow', 'product-sale-FE'])
        origin = origin_div.text.strip() if origin_div else "Origin not found"

        product_tuple = (title, origin, name, price)
        if product_tuple not in unique_products:
            unique_products.add(product_tuple)
            products.append({
                'name': title,
                'destination': origin,
                'seller': name,
                'price': price
            })
    return products

def crawler_exec(version):
    """
    Crawls a specific page version of the market and saves the HTML to disk.
    """
    if version == 1:
        url = "http://elysiumpir7gp5j7pl3kye36xnj5zig4fe5ng67ji7s5kahj3jiitfqd.onion/category/289"
    else:
        url = "http://elysiumpir7gp5j7pl3kye36xnj5zig4fe5ng67ji7s5kahj3jiitfqd.onion/category/289?page=" + str(version)
    
    # Session cookies for authentication
    cookies = {
        "elysium_session": "eyJpdiI6IkkzalVlTk8zQWM1K09GS1dPY3AxY0E9PSIsInZhbHVlIjoiU1JrMlFHTlExRklkd0tsK2hZN29GUVZLQUVOcnB0ZllUWTVzTHZZaG8wQzkvYXhaekVZU1NwZGxrK3F3Z3F0bTdVZFRYTUUxS3Z3ZFZOVytTcnpxRnBiZS8zeW50WERlV2FtZWxnaTJqOUVXdU1NMzJWNmJpY2JBWEU0dG5iNWQiLCJtYWMiOiJjMTk3MTIxZGZjMjE1ZTllYTY2ZTYwYjk1MTM0YjA2MjQxN2IzMmJiMWM3ODI1MzJhODFmYjk0YmQzZGRiNmNkIiwidGFnIjoiIn0%3D",
        "XSRF-TOKEN": "eyJpdiI6IkR2R2YzNEhwemkzaEFSS1FnT0t3b1E9PSIsInZhbHVlIjoiVGJaanMwOFBlMU1HYXRZRXBkQUg3cWZ6ejgwQmRka3VLcHZyUno3clNObDdWcjNSVmlRS0gzdTF3Rjl1YTNUZHBmOHM5VXVCQU9RZFF5bi8wMHh1dzFjOUdYNnlWMW1LQWFYZ3RqL0FySnFySHY0TGJtOWhBWXhxMDFaK2NuM00iLCJtYWMiOiI1NjY1MzU1NWEwMDJjYzQxMDIyYzExYjJlNDU1NWVhODhmYmJlNzUzZTFlMDc2NzBhYmNjZjIzNWU5YjM4OGRkIiwidGFnIjoiIn0%3D"
    }
    content = crawl_page(url, cookies)
    html_content = content.text
    html_path = "elysium" + str(version) + ".html"

    # Save the HTML content to a file
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
        print("wrote to html")

def scraper_exec(version):
    """
    Loads a saved HTML file and extracts product data.
    """
    html_path = "Sites/elysium" + str(version) + ".html"
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as html_file:
        html_content = html_file.read()
    products = scrape_products(html_content)
    return products

def extract_number_and_check_kg(name):
    """
    Extracts the first number from a string and converts to grams if 'kg' is present.
    Returns the number or None if not found.
    """
    numbers = re.findall(r'\d+', name)
    if numbers:
        number = int(numbers[0])
        if 'kg' in name:
            number *= 1000
        return number
    return None

if __name__ == '__main__':
    # Crawl and save multiple pages
    for i in range(1, 20):
        crawler_exec(i)
    all_drugs = []
    # Scrape product data from saved HTML files
    for j in range(1, 20):
        new_products = scraper_exec(j)
        print(len(new_products))
        all_drugs.append(new_products)
    # Flatten the list of product lists
    flattened_data = [item for sublist in all_drugs for item in sublist]
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(flattened_data)
    # df['amount_of_drugs'] = df['name'].apply(extract_number_and_check_kg)
    df.to_csv('elysium.csv')




