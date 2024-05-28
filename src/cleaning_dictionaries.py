import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')


# contractions
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"di": "hindi",
"kundi": "kung hindi"
}


# stop words
# source: https://github.com/explosion/spaCy/blob/master/spacy/lang/tl/stop_words.py
tagalog_stop_words = set(
    """
      akin
      aking
      ako
      alin
      am
      amin
      aming
      ang
      ano
      anumang
      apat
      at
      atin
      ating
      ay
      bababa
      bago
      bakit
      bawat
      bilang
      dahil
      dalawa
      dapat
      din
      dito
      doon
      gagawin
      gayunman
      ginagawa
      ginawa
      ginawang
      gumawa
      gusto
      habang
      hanggang
      hindi
      huwag
      iba
      ibaba
      ibabaw
      ibig
      ikaw
      ilagay
      ilalim
      ilan
      inyong
      isa
      isang
      itaas
      ito
      iyo
      iyon
      iyong
      ka
      kahit
      kailangan
      kailanman
      kami
      kanila
      kanilang
      kanino
      kanya
      kanyang
      kapag
      kapwa
      karamihan
      katiyakan
      katulad
      kaya
      kaysa
      ko
      kong
      kulang
      kumuha
      kung
      laban
      lahat
      lamang
      likod
      lima
      maaari
      maaaring
      maging
      mahusay
      makita
      marami
      marapat
      masyado
      may
      mayroon
      mga
      minsan
      mismo
      mula
      muli
      na
      nabanggit
      naging
      nagkaroon
      nais
      nakita
      namin
      napaka
      narito
      nasaan
      ng
      ngayon
      ni
      nila
      nilang
      nito
      niya
      niyang
      noon
      o
      pa
      paano
      pababa
      paggawa
      pagitan
      pagkakaroon
      pagkatapos
      palabas
      pamamagitan
      panahon
      pangalawa
      para
      paraan
      pareho
      pataas
      pero
      pumunta
      pumupunta
      sa
      saan
      sabi
      sabihin
      sarili
      sila
      sino
      siya
      tatlo
      tayo
      tulad
      tungkol
      una
      walang
      gcash
      app
      apps
      still
      mag
      really
      nyo
      mo
      po
      nmn
      im
      niyo
      ba
      nag
      yung
      lang
      though
      CENSORED
      something
      opo
      pag
      mo
      let
      star
      yan
      kayo
      nga
      ba
      ung
      naka
      let
      nalang
      naman
      sana
      lng
      maka
      pls
      thank
      please
      cause
      another
      mag
      sometime
      since
      yet
      also
      much
      every
      us
      lot
      nman
      nang
      sana
      wow
      many
      thing
      but
      nang
      well
      mas
      dont
      ye
      hi
      verry
      lalo
      talaga
      tlg
      tlga
      well
      bless
      ok
      okay
      okey
      tapos
      nakaka
      rin
      anymore
      ayo
      dam
      eh
      parin
      kwenta
      almost
      kasi
      especially
      become
      pang
      sya
      tapos
      kaso
      yun
      come
      put
      even
      see
      happen
      kc
      globe
      akong
      kasi
      make
      bat
      bkit
      5star
      see
      agad
      ap
  """.split()
)

# english stopwords
stopwords_list = stopwords.words('english')
stopwords_list.extend(tagalog_stop_words)
stopwords_list.extend(['..', '...'])


# bad words
tagalog_bad_words = {
"bwesit",
"buluk",
"bulok",
"putanginang",
"amputa",
"animal ka",
"animal",
"bilat",
"binibrocha",
"bobo",
"bogo",
"boto",
"brocha",
"burat",
"bwesit",
"bwisit",
"demonyo ka",
"demonyo",
"engot",
"etits",
"gaga",
"gagi",
"gago",
"habal",
"hayop",
"hayup",
"hinampak",
"hinayupak",
"hindot",
"hindutan",
"hudas",
"iniyot",
"inutel",
"inutil",
"iyot",
"ina",
"kagaguhan",
"kagang",
"kantot",
"kantotan",
"kantut",
"kantutan",
"kaululan",
"kayat",
"kiki",
"kikinginamo",
"kingina",
"kupal",
"leche",
"leching",
"lechugas",
"lintik",
"nakakaburat",
"nimal",
"ogag",
"olok",
"pakingshet",
"pakshet",
"pakyu",
"pesteng yawa",
"poke",
"poki",
"pokpok",
"poyet",
"pu'keng",
"pucha",
"puchanggala",
"puchangina",
"puke",
"puki",
"pukinangina",
"puking",
"punyeta",
"puta",
"putang",
"putang ina",
"putangina",
"putanginamo",
"putaragis",
"putragis",
"puyet",
"ratbu",
"shunga",
"shuta",
"sira ulo",
"sira",
"siraulo",
"suso",
"susu",
"tae",
"taena",
"tamod",
"tanga",
"tangina",
"taragis",
"tarantado",
"tete",
"teti",
"timang",
"tinil",
"tite",
"titi",
"tungaw",
"ulol",
"ulul",
"ungas",
"empyerno",
"putanginq",
"yudipota",
"bilatsangiloy",
"blablabla",
"gagago"
}

english_profanity = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/profanity_wordlist.txt", header=None)
profanity_list = english_profanity[0].tolist()
tagalog_bad_words.update(profanity_list) # update - set and dictionaries, extend - lists