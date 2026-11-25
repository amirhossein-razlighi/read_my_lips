"""Span rules shared by every silenced-span test set.

A silenced span must never start or end with a stop word or a filler: WhisperX timings of short unstressed words
and back-channels ("a", "yeah") are the least reliable, so such an edge word can stay audible while the reference
counts it as silenced.
"""
import re

# stop words of the window builder (per language)
STOPWORDS = {
    "en": set(["a", "an", "the", "and", "or", "but", "so", "of", "to", "in", "on", "at", "for", "with", "from", "by", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we", "you", "i", "me", "him", "her", "them", "us", "my", "your", "his", "our", "their", "there", "here", "not", "no", "yes", "then", "than", "as", "if", "just", "very", "really", "also"]),
    "es": set(["el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "de", "del", "a", "al", "en", "con", "por", "para", "que", "es", "son", "era", "fue", "ser", "estar", "esta", "este", "esto", "esos", "esas", "mi", "tu", "su", "nos", "les", "lo", "le", "se", "yo", "tú", "él", "ella", "ellos", "ellas", "nosotros", "usted", "ustedes", "no", "sí", "muy", "más", "como", "si", "cuando", "donde", "porque", "también", "ya", "hay"]),
    "fr": set(["le", "la", "les", "un", "une", "des", "et", "ou", "mais", "de", "du", "au", "aux", "en", "dans", "sur", "avec", "par", "pour", "que", "qui", "est", "sont", "était", "été", "être", "ce", "cette", "ces", "cet", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on", "ne", "pas", "non", "oui", "très", "plus", "comme", "si", "quand", "où", "parce", "aussi", "déjà", "y", "a", "d", "l", "j", "n", "c", "s", "qu"]),
    "ru": set(["и", "в", "во", "не", "на", "я", "он", "она", "оно", "они", "мы", "вы", "ты", "что", "это", "эта", "этот", "эти", "то", "та", "тот", "те", "с", "со", "как", "а", "но", "да", "или", "же", "бы", "ли", "у", "к", "ко", "от", "до", "по", "за", "из", "под", "над", "о", "об", "обо", "для", "при", "про", "без", "между", "через", "был", "была", "было", "были", "есть", "быть", "будет", "нет", "ещё", "еще", "уже", "вот", "там", "тут", "здесь", "так", "очень", "тоже", "также", "когда", "где", "потому", "если", "чем", "мой", "моя", "мое", "твой", "его", "её", "ее", "их", "наш", "ваш", "свой", "себя", "меня", "тебя", "нас", "вас", "них", "себе", "мне", "тебе", "нам", "вам", "им", "ей", "ему", "них", "всё", "все", "весь", "вся"]),
}

FILLERS = {
    "en": set(["yeah", "yes", "yep", "yup", "okay", "ok", "oh", "uh", "um", "umm", "uhm", "hmm", "mm", "mhm", "ah", "er", "eh", "like", "well", "so", "right"]),
    "es": set(["eh", "este", "pues", "bueno", "vale", "ah", "mm", "mmm", "sea", "ok", "okay"]),
    "fr": set(["euh", "ben", "bon", "alors", "voilà", "voila", "ah", "hein", "quoi", "bah", "ouais", "ok", "okay"]),
}


def norm(w: str) -> str:
    return re.sub(r"[^\w']+", "", w.lower()).strip("'")


def edge_ok(words, language: str) -> bool:
    """True when neither the first nor the last silenced word is a stop word, a filler, or a single character."""
    if not words:
        return False
    bad = STOPWORDS.get(language, set()) | FILLERS.get(language, set())
    for w in (words[0], words[-1]):
        n = norm(w)
        if len(n) < 2 or n in bad:
            return False
    return True
