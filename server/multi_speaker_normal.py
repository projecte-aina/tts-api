import argparse
import asyncio
import io
import json
import os
import sys
import traceback
import time
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Union

import torch
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket

from utils.argparse import MpWorkersAction

torch.set_num_threads(1)

# global path variables
path = Path(__file__).parent / ".models.json"
path_dir = os.path.dirname(path)
manager = ModelManager(path)

# default tts model/files
models_path_rel = '../models/vits_ca'
model_ca = os.path.join(path_dir, models_path_rel, 'best_model.pth')
config_ca = os.path.join(path_dir, models_path_rel, 'config.json')

def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path",
                        default=config_ca,
                        type=str,
                        help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_ca,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=8000, help="port to listen on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip to listen.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--mp_workers", action=MpWorkersAction, type=int, default=2,
                        help="number of CPUs used for multiprocessing")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    parser.add_argument("--speech_speed", type=float, default=1.0, help="Change speech speed.")
    return parser


def update_config(config_path, velocity):
    length_scale = 1 / velocity
    with open(config_path, "r+") as json_file:
        data = json.load(json_file)
        data["model_args"]["length_scale"] = length_scale
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()


# parse the args
args = create_argparser().parse_args()

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None
# new_speaker_ids = None
# use_aliases = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path
    speaker_ids_path = os.path.join(path_dir, models_path_rel, 'speaker_ids.json')

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# CASE4: change speaker speed
if args.speech_speed != 1.0:
    update_config(config_path, args.speech_speed)


syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=speakers_file_path,
        tts_languages_file=None,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=args.use_cuda,
    )

new_speaker_ids = json.load(open(speaker_ids_path))

input_speaker_id = new_speaker_ids["f_cen_81"]

text1 = "L'exploració espacial és una proesa de la humanitat que va més enllà dels límits de la nostra existència terrenal, penetrant en l'infinit misteri que és l'univers. Aquest viatge, que ha captivat la nostra imaginació des de temps immemorials, és tan una aventura científica com una reflexió profunda sobre qui som i quin lloc ocupem en el vast univers. L'interès per l'espai no és res de nou. Des de l'antiga Grècia, els astrònoms han contemplat les estrelles amb fascinació i curiositat. L'observació de l'espai ha evolucionat al llarg dels segles, però no va ser fins al segle XX que l'humanitat va aconseguir l'extraordinari: sortir de la Terra i viatjar més enllà de la seva atmosfera. La carrera espacial, començada al final de la Segona Guerra Mundial, va establir les bases d'aquesta extraordinària aventura. El 1957, l'URSS va llançar el Sputnik, el primer satèl·lit artificial. Aquest esdeveniment va marcar un punt d'inflexió, iniciant una nova era d'exploració i descoberta. La competència entre l'Est i l'Oest per dominar l'espai va esdevenir un símbol de la Guerra Freda, però alhora va accelerar els avenços tecnològics i científics.Després del Sputnik, el següent gran pas va ser la missió Apollo 11 de la NASA. El 20 de juliol de 1969, Neil Armstrong va fer història en posar el peu a la Lluna, marcat pel seu famós comentari: Aquest és un petit pas per a un home, però un gran salt per a la humanitat . Aquell moment va capturar la imaginació del món i va reafirmar la nostra capacitat per aconseguir grans coses. Des d'aleshores, l'exploració espacial ha continuat amb una sèrie d'innovacions i descobertes. Hem enviat sondes a Mars, Venus i més enllà, ens hem apropat als límits del nostre sistema solar i hem observat l'univers com mai abans amb telescopis espacials com el Hubble. Però l'espai no és només un lloc per a la ciència; també és un espai per a la reflexió. Ens fa preguntar-nos sobre la nostra posició en l'univers, sobre l'origen i el futur de la vida i sobre el nostre destí com a espècie. Els avenços en astrobiologia, per exemple, ens estan portant més a prop de respondre a la pregunta de si estem sols en l'univers. Actualment, ens trobem en una nova era d'exploració espacial. Amb el desenvolupament de tecnologies de coets reutilitzables i projectes com Artemis de la NASA, que pretén retornar humans a la Lluna, i les missions de SpaceX a Mart, el futur de l'exploració espacial és emocionant. Però, malgrat tot això, l'espai segueix sent un lloc perillós i inhòspit. Cada missió espacial implica un risc considerable i exigeix una planificació i una preparació minucioses. Els desafiaments tecnològics, humans i ètics que planteja l'exploració espacial són enormes. Tot i així, continuem esforçant-nos per arribar més lluny. Per què? Perquè l'espai, amb la seva bellesa i misteri, ens crida. Ens crida a explorar, a aprendre i a comprendre. En el seu immens silenci, l'espai ens parla de nosaltres mateixos, del nostre lloc en l'univers i del nostre potencial per aconseguir grans coses. L'exploració espacial és, en última instància, un viatge d'autodescobriment. A mesura que avancem cap al futur, podem estar segurs que l'exploració espacial continuarà. Continuarem estirant els límits del que és possible, afrontant els desafiaments i buscant respostes a les nostres preguntes més profundes. I en aquesta recerca, potser trobarem no només nous mons, sinó també noves maneres de veure i entendre a nosaltres mateixos."

start_time = time.time()
outputs1 = syn.tts(text1, input_speaker_id)
end_time = time.time()
print(f"Time taken for inference 1: {end_time - start_time} seconds")
syn.save_wav(outputs1, "normal_1.wav")

text2 = "La dieta mediterrània és més que una simple llista d'aliments a consumir o evitar. És un estil de vida que abasta una alimentació equilibrada, exercici regular i una actitud positiva cap a la vida. Encarnada en els costums i tradicions de la gent que viu a les costes del Mediterrani, aquesta dieta ha estat reconeguda per la UNESCO com a Patrimoni Cultural Inmaterial de la Humanitat, i per un bon motiu.En primer lloc, cal entendre que la dieta mediterrània no és una dieta en el sentit restrictiu del terme. No es tracta de comptar calories o de pesar aliments, sinó de menjar una varietat d'aliments frescos i naturals en quantitats moderades. Aquesta filosofia d'alimentació equilibrada, combinada amb l'activitat física regular, promou un estil de vida saludable i actiu. L'essència de la dieta mediterrània es basa en el consum d'aliments plant-based. Això inclou fruites i verdures, llegums, fruits secs, cereals integrals i olives. El peix i les carns blanques es mengen en moderació, mentre que les carns vermelles es limiten. També és destacable l'ús de l'oli d'oliva verge extra com a principal font de greixos saludables. Un dels elements més característics de la dieta mediterrània és l'ús abundant d'herbes i espècies per a aromatitzar els aliments, en comptes de sal en excés o greixos saturats. Això no només aporta sabors rics i variats, sinó que també ofereix beneficis per a la salut gràcies a les propietats antioxidants i antiinflamatòries d'aquests condiments naturals. El vi és un altre component clau de la dieta mediterrània, consumit amb moderació, normalment amb els àpats. Això es deu al seu contingut en resveratrol, un potent antioxidant. Però és important recordar que el consum d'alcohol ha de ser responsable i no és necessari per gaudir dels beneficis de la dieta mediterrània. Més enllà de l'alimentació, la dieta mediterrània implica una actitud cap a la vida que valorisa la convivència i la disfrutació dels àpats en companyia. Es tracta de prendre's el temps per menjar, gaudir dels aliments i compartir moments amb la família i els amics. Diversos estudis científics han demostrat els beneficis de la dieta mediterrània per a la salut. Aquests inclouen una millor salut cardiovascular, una reducció del risc de diabetis tipus 2, una millor salut mental i un augment de la longevitat. També s'ha relacionat amb un menor risc de malalties neurodegeneratives com l'Alzheimer. La dieta mediterrània és, en última instància, una celebració del plaer de menjar, del respecte per l'entorn i del gaudi de la vida. Amb la seva rica varietat d'aliments nutritius i deliciosos i la seva èmfasi en l'equilibri i la moderació, ofereix una guia invaluable per a un estil de vida saludable i satisfactori."

start_time = time.time()
outputs2 = syn.tts(text2, input_speaker_id)
end_time = time.time()
print(f"Time taken for inference 2: {end_time - start_time} seconds")
syn.save_wav(outputs2, "normal_2.wav")

text3 = "El primer viatge espacial humà és un capítol brillant en el llibre de la història humana. L'epopeia que va posar el cosmonauta rus Yuri Gagarin al firmament de l'eternitat es va iniciar el 12 d'abril de 1961, quan es va convertir en el primer ésser humà a viatjar a l'espai. L'època era la del fervor de la Guerra Freda, un període de rivalitat intensa entre l'Orient socialista i l'Occident capitalista. L'espai es va convertir en el camp de batalla suprem per demostrar la superioritat tecnològica. La Unió Soviètica i els Estats Units competien per aconseguir fites espacials, en un episodi conegut com a carrera espacial. Gagarin, un jove pilot de l'aire de 27 anys, va ser seleccionat per a la missió Vostok 1 després d'un rigorós i intens procés de selecció. La seva personalitat amable, el seu somriure captivador i el seu estatus de fill de treballadors agrícoles el van convertir en l'elecció ideal per a la propaganda soviètica. La missió va ser envoltada de secretisme. Inclús Gagarin no va saber que seria l'elegit fins uns dies abans del llançament. La seva missió: orbitar la Terra una vegada i tornar sa i estalvi. El 12 d'abril de 1961, amb Gagarin abord, la nau espacial Vostok es va llançar des de Baikonur, Kazakhstan. Quan el coet va arrencar, Gagarin va exclamar les seves famoses paraules Poyekhali! - Anem!. Durant 108 minuts, Gagarin va orbitar la Terra, convertint-se en el primer ésser humà a veure el planeta des de l'espai. La seva descripció de la Terra com un planeta blau amb núvols blancs va captivar la imaginació del món. Després d'una reentrada tensa, durant la qual la càpsula espacial Vostok es va incendiar breument, Gagarin va aterrar amb èxit a una granja a prop de Saratov, Rússia. Encara que el seu paracaigudes va fer que aterres a algunes milles del lloc previst, va ser rebut com un heroi per la família de la granja. La notícia de la seva èpica jornada va recórrer el món. Gagarin va ser aclamat com un heroi, no només a la Unió Soviètica, sinó també internacionalment. La seva façana somrient i humil va captivar a la gent de tot el món, i la seva façana va aparèixer a la portada de revistes de tot el món. Però, més enllà de la propaganda i la política, el primer viatge espacial de Gagarin va ser un moment clau en la nostra història com a espècie. Per primera vegada, un ésser humà va haver superat els límits de la nostra atmosfera i havia contemplat la Terra des de l'espai. Aquesta proesa va obrir la porta a tota una nova era d'exploració espacial, demostrant que l'espai no era un lloc inaccessible per a nosaltres. La valentia de Gagarin, la seva humilitat i la seva intrèpida curiositat personifiquen l'esperit d'exploració humana. Encara que la seva vida va ser curta - va morir en un accident d'avió als 34 anys - el seu llegat continua inspirant a les generacions futures a mirar cap a les estrelles i somiar amb el que és possible."

start_time = time.time()
outputs3 = syn.tts(text3, input_speaker_id)
end_time = time.time()
print(f"Time taken for inference 3: {end_time - start_time} seconds")
syn.save_wav(outputs3, "normal_3.wav")
