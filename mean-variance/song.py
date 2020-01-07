import numpy as np
class Song(object):

    def getNotes(self):
        bars = self.notes.split('|')
        songNotes = []
        for bar in bars:
            for notes in bar:
                for note in notes:
                    if not note == '' and not note == " ":
                        songNotes.append(note)
        self.songNotes = np.array(songNotes)
        self.uniqueNotes = np.array(np.unique(self.songNotes))
        return self.songNotes

    def getBars(self):
        return self.notes.split('|')

    def getSong(self):
        return self.metric + " " + self.songKey +" "+ self.notes

    """docstring for Song."""
    def __init__(self):
        self.title = None
        self.metric = None
        self.songKey = None
        self.notes = None
        self.songNotes = None
        self.uniqueNotes = None
