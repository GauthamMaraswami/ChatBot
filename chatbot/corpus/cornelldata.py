
import os
import ast


class CornellData:
 
    def __init__(self, dirName):

        self.lines = {}
        self.conversations = []

        MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]

        self.lines = self.loadLines(os.path.join(dirName, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        self.conversations = self.loadConversations(os.path.join(dirName, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)

      

    def loadLines(self, fileName, fields):

        lines = {}

        with open(fileName, 'r', encoding='iso-8859-1') as f:  
            for line in f:
                values = line.split(" +++$+++ ")

                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self, fileName, fields):

        conversations = []

        with open(fileName, 'r', encoding='iso-8859-1') as f:  
            for line in f:
                values = line.split(" +++$+++ ")

                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def getConversations(self):
        return self.conversations
