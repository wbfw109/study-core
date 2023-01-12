import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class PythonOrgSearch(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome("rsrc\chromedriver_win32_83.0.4103.39.exe")

    # This is the test case method. The test case method should always start with characters test.
    def test_search_in_python_org(self):
        driver = self.driver

        # The driver.get method will navigate to a page given by the URL. WebDriver will wait until the page has fully loaded (that is, the “onload” event has fired) before returning control to your test or script. It’s worth noting that if your page uses a lot of AJAX on load then WebDriver may not know when it has completely loaded.:
        driver.get("http://www.python.org")
        self.assertIn("Python", driver.title)
        elem = driver.find_element_by_name("q")
        elem.send_keys("pycon")
        elem.send_keys(Keys.RETURN)
        assert "No results found." not in driver.page_source

    def tearDown(self):
        # You can also call quit method instead of close. The quit will exit the entire browser, whereas close will close a tab, but if it is the only tab opened, by default most browser will exit entirely.:
        self.driver.close()


if __name__ == "__main__":
    unittest.main()
