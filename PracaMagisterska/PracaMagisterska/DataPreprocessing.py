import re

class Preprocessing:

    @staticmethod
    def deleteHtmlTags(text):
        return re.sub('<[^>]*>', '', text)

    @staticmethod
    def deleteNonWordChars(text):
        return re.sub('[\W]+', ' ', text.lower())

    @staticmethod
    def deleteUrls(text):
        return re.sub('(https{0,1}://).[^\s]+', '', text)

    def deleteHtmlEntities(text):
        return re.sub('&[#]{0,1}\w{1,}[;]', '', text);