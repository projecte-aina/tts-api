import pytest
import json
from fastapi import status
from fastapi.testclient import TestClient
from server.tests.base_test_case import APIBaseTestCase


class TestApi(APIBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_text_to_voice(self):
        options = {
            "voice": "f_cen_095",
            "type": "text",
            "text": "hola"
        }
        with TestClient(self.app) as client:
            response = client.post(url="/api/tts", content=json.dumps(options))

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert response.content is not None
 
    def test_text_to_voice_error(self):
        msg = {
            "message":"sfsfs is an unknown speaker id.",
            "accept":["f_cen_095","f_cen_092","m_occ_072","m_cen_pau","m_occ_7b7","m_val_51c","f_cen_063","f_cen_051"]
            }
        options = {
            "voice": "sfsfs",
            "type": "text",
            "text": "hola"
        }
        with TestClient(self.app) as client:
            response = client.post(url="/api/tts", content=json.dumps(options))

        assert response.status_code == status.HTTP_406_NOT_ACCEPTABLE
        content = json.loads(response.content)
        assert content["message"] == msg["message"]
        assert content["accept"] == msg["accept"]

    
    def test_list_voices(self):

        with TestClient(self.app) as client:
            response = client.get(url="/api/available-voices")

        assert response.status_code == status.HTTP_200_OK
        voices = json.loads(response.content)["voices"]
        assert voices == ["f_cen_095","f_cen_092","m_occ_072","m_cen_pau","m_occ_7b7","m_val_51c","f_cen_063","f_cen_051"]


    @pytest.mark.repeat(25)
    def test_multi_larger_text(self):
        options = {
            "voice": "f_cen_095",
            "type": "text",
            "text": ""
        }

        for sentence in texts:
            options["text"] = sentence

            with TestClient(self.app) as client:
                response = client.post(url="/api/tts", content=json.dumps(options))

            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
            assert response.content is not None
        

texts = [
    "Els vents del nord bufen amb ferocitat, sacsejant els arbres i fent xiular les finestres dels casals. Els habitants de Sant Miquel de la Vall s'abriguen amb abrics gruixuts i es refugien al caliu dels seus llars, esperant que la tempesta passi. Les muntanyes que envolten el poble es converteixen en una estampa impressionant, amb els cims nevats brillant sota la llum de la lluna. Malgrat el temps advers, la vida al poble continua amb la seva rutina habitual. Els pagesos treballen les terres, preparant-les per a la nova temporada de cultius. Els artesans confeccionen objectes amb habilitat i destresa, mantenint viva la tradició ancestral del poble. Els nens corretegen pels carrers, jugant amb la neu i gaudint de la seva infantesa sense preocupacions. A les tardes, els veïns es reuneixen a la taverna del poble per compartir una copa de vi i conversar sobre els esdeveniments del dia. Les rialles ressonen en les parets de pedra, creant una atmosfera de camaraderia i amistat. Amb l'arribada de la primavera, el paisatge es transforma en un mar de colors, amb les flors brotant de cada racó del poble. Els ocells tornen a cantar amb alegria, omplint l'aire amb el seu càntic melodios. Els habitants celebren l'arribada de la nova estació amb festes i celebracions, agraint la renovació de la vida i l'esperança que porta amb ella. Sant Miquel de la Vall continua sent un refugi per a tots aquells que busquen pau i tranquil·litat en un món en constant canvi."
    ,"Les campanes de l'església del poble repiquen amb alegria, cridant a tots els habitants a la missa del diumenge. Els fidels s'apressen a arribar, vestits amb les seves millors robes i amb somriures al rostre. El sacerdot pronuncia les paraules sagrades, recordant als fidels els valors de la fe i la bondat. Les veus del cor ressonen en l'espai sagrat, omplint-lo de música celestial. Després de la missa, els habitants es reuneixen a la plaça del poble per compartir un àpat frugal i conversar animadament sobre els esdeveniments de la setmana. Hi ha una atmosfera de calma i serenitat que impregna l'aire. Els nens juguen alegrement pels carrers, rient i cridant amb innocència. Els adults observen amb tendresa les seves cabrioles, recordant els dies en què ells mateixos eren joves i despreocupats. Amb el capvespre, el cel es tenyeix de colors pastel, creant un espectacle impressionant que captiva la mirada dels espectadors. Les orenetes volen en cercles al voltant del campanar, anunciant l'arribada de la nit. A mesura que la foscor envaeix el paisatge, les estrelles comencen a brillar amb intensitat en el firmament. Els habitants del poble es retiren a les seves llars, amb la sensació de pau i benestar que només es troba en el cor d'un lloc com Sant Miquel de la Vall."
    ,"la riquesa del territori, amb plats elaborats amb productes frescos i tradicionals que captiven els sentits. Els vespres d'estiu són màgics a Sant Miquel de la Vall, amb la gent reunita a les terrasses dels bars, gaudint de la frescor de la nit i de la companyia dels amics. Les converses flueixen lliurement, embolicades pel perfum de les flors i el so dels grills que canten a l'horabaixa. És en aquests moments que es crea el veritable esperit del poble, una barreja d'hospitalitat i calidesa que acull a tots els que s'hi acosten. Al matí, els primers raigs de sol il·luminen el paisatge, revelant la seva bellesa oculta i donant pas a un nou dia ple de possibilitats. Els habitants es desperten amb energia i entusiasme, preparats per viure les aventures que el futur els té reservades. Amb cada sortida i posta de sol, Sant Miquel de la Vall continua escrivint la seva història, una història que parla de superació, solidaritat i amor per la terra. És un lloc que captiva el cor i l'ànima, un lloc que mai s'oblida i al qual sempre es torna amb alegria."
    ,"Els matins a Sant Miquel de la Vall comencen amb el cant dels gallins i el soroll dels primers vehicles que es dirigeixen cap als camps. El sol es filtra entre les branques dels arbres, il·luminant suament els carrers tranquils del poble. Les olors de pa fresc i cafè recent torrat es barregen en l'aire, convidant els habitants a començar el dia amb energia i optimisme. Les botigues del poble obren les seves portes, i els comerciants preparen les seves parades amb productes frescos i artesanals. El mercat local es converteix en el centre de l'activitat, amb clients i venedors intercanviant salutacions i notícies mentre negocien les seves compres. Els nens van a l'escola amb les seves motxilles plenes d'il·lusions i llibres, preparats per aprendre i créixer amb cada nova experiència. Mentre tant, els treballadors de les oficines es preparen per a un dia de feina, revisant correus electrònics i fent trucades importants. El ritme tranquil i ordenat de la vida al poble es manté, amb cada persona fent la seva part per contribuir al benestar col·lectiu. A la nit, les llums dels llars brillen a través de les finestres, creant una estampa reconfortant de calidesa i seguretat enmig de la foscor. És un recordatori constant dels llaços forts que uneixen als habitants de Sant Miquel de la Vall, i de la seva capacitat per superar els desafiaments junts."
    ,"Les tardes a Sant Miquel de la Vall són tranquil·les i apacibles, amb els habitants del poble gaudint dels seus moments de lleure i descans. Alguns es passegen pels carrers empedrats, admirant l'arquitectura antiga i els detalls artesanals de les cases. Altres es reuneixen als parcs i jardins del poble, on es pot escoltar el xiuxiueig del vent entre els arbres i el cant dels ocells. Alguns habitants prefereixen quedar-se a casa i llegir un llibre o escoltar música, mentre que altres es dediquen a les seves aficions com la jardineria, la pintura o la cuina. Sigui quin sigui el seu passatemps preferit, tots els habitants de Sant Miquel de la Vall gaudeixen dels moments de pau i tranquil·litat que ofereix el poble. A mesura que el sol es pon i la nit cau sobre el paisatge, els llums del poble comencen a brillar, creant una atmosfera màgica i acollidora. És un moment perfecte per reunir-se amb amics i familiars i compartir una bona conversa i un got de vi. Malgrat les preocupacions i els reptes de la vida quotidiana, els habitants de Sant Miquel de la Vall troben sempre temps per apreciar les petites coses i gaudir de la companyia dels seus ser estimats."
    ,"Els colors de la tardor pinten el paisatge amb una paleta de tons càlids i vibrants. Les fulles dels arbres canvien del verd al groc, taronja i vermell, creant un mosaic de colors que captiva la mirada dels espectadors. Els vents suaus porten consigo l'olor de terra humida i la promesa de nous començaments. Les muntanyes que envolten el poble es vesteixen amb una capa de fulles caigudes, formant una catifa natural que crida a ser explorada. Els habitants del poble es preparen per a la temporada de recol·lecció, omplint les cistelles amb fruites i verdures de l'hort. Les festes de la tardor omplen el calendari, amb celebracions que honoren les tradicions i costums ancestrals. Les fogueres crepitants il·luminen les nits fresques, mentre els habitants del poble es reuneixen al seu voltant per compartir històries i rialles. És un temps de gratitud i reflexió, quan els habitants del poble donen gràcies per les seves benvingudes i esperen amb il·lusió els temps que estan per venir."
    ,"Les cases de pedra del poble s'alineen al llarg dels carrers empedrats, formant un laberint d'històries i records. Les parets porten marques del temps, amb pedres gastades i finestres adornades amb geranis de colors vius. Els balcons estan engalanats amb banderoles i catifes durant les festes locals, creant una imatge festiva que il·lumina el carrer. Al capvespre, les llums suaus dels fanals s'encenen, projectant ombres misterioses sobre les parets de pedra. Els veïns es reuneixen als balcons i les places del poble per gaudir de l'ambient tranquil i contemplar la bellesa del seu entorn. És en aquests moments tranquils que es poden sentir les veus del passat, recordant als habitants del poble la importància de la seva història i la seva cultura."
]