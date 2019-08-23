import re

class Preprocessing:

    @staticmethod
    def deleteHtmlTags(text):
        return re.sub(r'<[^>]*>', '', text)

    @staticmethod
    def deleteNonWordChars(text):
        return re.sub(r'[\W]+', ' ', text.lower())

    @staticmethod
    def deleteUrls(text):
        return re.sub(r'(https{0,1}://).[^\s]+', '', text)

    @staticmethod
    def deleteHtmlEntities(text):
        return re.sub(r'&[#]{0,1}\w{1,}[;]', '', text)

    @staticmethod
    def deletePunctuation(text):
        return re.sub(r'[.!,;?]', ' ', text)

    @staticmethod
    def deleteMultipleSpaces(text):
        return re.sub(r' +', ' ', text)