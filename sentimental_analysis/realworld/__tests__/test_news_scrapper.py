import unittest
from unittest.mock import patch, MagicMock
from realworld.newsScraper import scrapNews, getNewsResults

class TestNewsScraper(unittest.TestCase):

    @patch('realworld.newsScraper.requests.get')
    @patch('realworld.newsScraper.BeautifulSoup')
    def test_getNewsResults_returns_urls(self, mock_soup, mock_get):
        # Mock the HTML content and parsing
        mock_get.return_value.content = '<html></html>'
        mock_soup.return_value.select.return_value = [
            MagicMock(find=MagicMock(return_value={'href': 'https://news.example.com/article1'})),
            MagicMock(find=MagicMock(return_value={'href': 'https://news.example.com/article2'})),
            MagicMock(find=MagicMock(return_value={'href': 'https://news.example.com/article3'}))
        ]

        result = getNewsResults('test query', 3)
        self.assertEqual(len(result), 3)
        self.assertIn('https://news.example.com/article1', result)

    @patch('realworld.newsScraper.requests.get')
    def test_getNewsResults_handles_no_results(self, mock_get):
        # Mock no content returned
        mock_get.return_value.content = '<html></html>'
        result = getNewsResults('test query', 0)
        self.assertEqual(result, [])

    # disabled due to cache rework
    # TODO(kurt): fix and re-enable
    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_uses_cache(self, mock_cache):
    #     # Mock cache returning data
    #     mock_cache.get.return_value = [{'Summary': 'Cached summary'}]
    #     result = scrapNews('test topic', 1, False)
    #     self.assertEqual(result, [{'Summary': 'Cached summary'}])

    # @patch('realworld.newsScraper.news_cache')
    # @patch('realworld.newsScraper.getNewsResults')
    # def test_scrapNews_calls_getNewsResults_when_no_cache(self, mock_getNewsResults, mock_cache):
    #     # Mock cache miss
    #     mock_cache.get.return_value = None
    #     mock_getNewsResults.return_value = []
    #     result = scrapNews('test topic', 1)
    #     self.assertEqual(result, [])
    #     mock_getNewsResults.assert_called_once_with('test topic', 20)

    # @patch('realworld.newsScraper.Article')
    # @patch('realworld.newsScraper.getNewsResults')
    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_processes_articles(self, mock_cache, mock_getNewsResults, mock_article_class):
    #     mock_cache.get.return_value = None
    #     mock_getNewsResults.return_value = ['https://news.example.com/article1']

    #     # Mock Article behavior
    #     mock_article = MagicMock()
    #     mock_article.text = 'This is a valid article text that is longer than 250 characters. ' * 5
    #     mock_article.summary = 'This is the article summary.'
    #     mock_article.download = MagicMock()
    #     mock_article.parse = MagicMock()
    #     mock_article.nlp = MagicMock()
    #     mock_article_class.return_value = mock_article

    #     result = scrapNews('test topic', 1)
    #     self.assertEqual(len(result), 1)
    #     self.assertEqual(result[0]['Summary'], 'This is the article summary.')

    # @patch('realworld.newsScraper.Article')
    # @patch('realworld.newsScraper.getNewsResults')
    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_skips_short_articles(self, mock_cache, mock_getNewsResults, mock_article_class):
    #     mock_cache.get.return_value = None
    #     mock_getNewsResults.return_value = ['https://news.example.com/article1']

    #     # Mock Article with short text
    #     mock_article = MagicMock()
    #     mock_article.text = 'Short text'
    #     mock_article.download = MagicMock()
    #     mock_article.parse = MagicMock()
    #     mock_article_class.return_value = mock_article

    #     result = scrapNews('test topic', 1)
    #     self.assertEqual(len(result), 0)

    # @patch('realworld.newsScraper.Article')
    # @patch('realworld.newsScraper.getNewsResults')
    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_handles_exceptions(self, mock_cache, mock_getNewsResults, mock_article_class):
    #     mock_cache.get.return_value = None
    #     mock_getNewsResults.return_value = ['https://news.example.com/article1']

    #     # Mock Article that raises an exception
    #     mock_article_class.side_effect = Exception('Download error')

    #     result = scrapNews('test topic', 1)
    #     self.assertEqual(result, [])

    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_saves_to_cache(self, mock_cache):
    #     mock_cache.get.return_value = None
    #     with patch('realworld.newsScraper.getNewsResults') as mock_getNewsResults, \
    #          patch('realworld.newsScraper.Article') as mock_article_class:
    #         mock_getNewsResults.return_value = ['https://news.example.com/article1']
    #         mock_article = MagicMock()
    #         mock_article.text = 'This is a valid article text that is longer than 250 characters. ' * 5
    #         mock_article.summary = 'This is the article summary.'
    #         mock_article.download = MagicMock()
    #         mock_article.parse = MagicMock()
    #         mock_article.nlp = MagicMock()
    #         mock_article_class.return_value = mock_article

    #         scrapNews('test topic', 1)
    #         mock_cache.set.assert_called_once()

    # @patch('realworld.newsScraper.news_cache')
    # def test_scrapNews_writes_json_output(self, mock_cache):
    #     mock_cache.get.return_value = [{'Summary': 'Cached summary'}]
    #     with patch('builtins.open', new_callable=MagicMock()) as mock_file:
    #         scrapNews('test topic', 1, jsonOutput=True)
    #         mock_file.assert_called_with('sentimental_analysis/realworld/news.json', 'w')

    # @patch('realworld.newsScraper.news_cache')
    # @patch('realworld.newsScraper.getNewsResults')
    # def test_scrapNews_handles_zero_articles(self, mock_getNewsResults, mock_cache):
    #     mock_cache.get.return_value = None
    #     mock_getNewsResults.return_value = []
    #     result = scrapNews('test topic', 0)
    #     self.assertEqual(result, [])

    def test_getNewsResults_with_invalid_query(self):
        # Assuming that getNewsResults will handle exceptions internally and return an empty list
        result = getNewsResults('', 3)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
