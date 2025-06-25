def sound(nastroenie):
    import pygame
    if nastroenie == "веселое" or nastroenie == "Веселое" or nastroenie == "Весёлое" or nastroenie == "весёлое":
        vosproizvedenie("NTR")
    elif nastroenie == "грусное" or nastroenie == "Грусное":
        vosproizvedenie("Сергей_Лазарев-Снег_в_океане")
    else:
        vosproizvedenie("kriticheskaya-oshibka-sistemnyiy-zvuk-windows-xp-23019")
    return None


def vosproizvedenie(name):
        import pygame
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("music/"+name+".mp3")
        pygame.mixer.music.play()
        return None 