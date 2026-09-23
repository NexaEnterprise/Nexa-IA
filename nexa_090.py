import os
import re
import json
import random
import unicodedata
from datetime import datetime

import numpy as np


# ============================================================
# NEXA AI 0.9
# Núcleo neural conversacional local
# Sem API externa
# ============================================================

NOME = "NEXA"
VERSAO = "0.9"

BASE_DIR = "nexa_data"

ARQ_CORPUS = os.path.join(
    BASE_DIR,
    "corpus.json"
)

ARQ_MEMORIA = os.path.join(
    BASE_DIR,
    "memoria.json"
)

ARQ_VOCABULARIO = os.path.join(
    BASE_DIR,
    "vocabulario.json"
)

ARQ_MODELO = os.path.join(
    BASE_DIR,
    "modelo.npz"
)

ARQ_CONFIG = os.path.join(
    BASE_DIR,
    "config.json"
)

os.makedirs(
    BASE_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

CONFIG_PADRAO = {

    "dimensao": 48,

    "oculta": 96,

    "taxa": 0.005,

    "epocas": 300,

    "temperatura": 0.75,

    "max_tokens": 40,

    "contexto": 16,

    "grad_clip": 5.0

}


def carregar_config():

    if not os.path.exists(
        ARQ_CONFIG
    ):

        salvar_config(
            CONFIG_PADRAO
        )

        return CONFIG_PADRAO.copy()

    try:

        with open(
            ARQ_CONFIG,
            "r",
            encoding="utf-8"
        ) as f:

            dados = json.load(f)

        config = CONFIG_PADRAO.copy()

        config.update(dados)

        return config

    except Exception:

        return CONFIG_PADRAO.copy()


def salvar_config(config):

    with open(
        ARQ_CONFIG,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4
        )


CONFIG = carregar_config()


# ============================================================
# UTILIDADES
# ============================================================

def normalizar(texto):

    texto = str(
        texto
    ).lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = re.sub(
        r"[^a-z0-9!?.,:;'\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def tokenizar(texto):

    return re.findall(
        r"\w+|[!?.,:;']",
        normalizar(texto)
    )


# ============================================================
# CORPUS INICIAL
# ============================================================

CORPUS_INICIAL = [

    ["oi", "Olá! Como posso ajudar?"],

    ["ola", "Olá! Eu sou a NEXA."],

    ["bom dia",
     "Bom dia! Como posso ajudar?"],

    ["boa tarde",
     "Boa tarde! Como posso ajudar?"],

    ["boa noite",
     "Boa noite! Como posso ajudar?"],

    ["qual seu nome",
     "Meu nome é NEXA."],

    ["como voce se chama",
     "Eu me chamo NEXA."],

    ["quem e voce",
     "Eu sou a NEXA, uma inteligência artificial conversacional."],

    ["o que e python",
     "Python é uma linguagem de programação muito utilizada no desenvolvimento de software."],

    ["para que serve python",
     "Python pode ser utilizado em inteligência artificial, automação, desenvolvimento web e análise de dados."],

    ["o que e inteligencia artificial",
     "Inteligência artificial é uma área da computação dedicada à criação de sistemas capazes de realizar tarefas complexas."],

    ["o que e ia",
     "IA significa inteligência artificial."],

    ["voce e uma ia",
     "Sim. Eu sou uma inteligência artificial desenvolvida localmente."],

    ["voce aprende",
     "Posso aprender novos exemplos através do meu treinamento."],

    ["voce tem memoria",
     "Sim. Possuo memória local para armazenar o histórico das conversas."],

    ["como voce funciona",
     "Eu utilizo redes neurais e processamento de linguagem para analisar texto."],

    ["voce usa api",
     "Não. Esta versão funciona localmente e não utiliza APIs externas."],

    ["obrigado",
     "De nada! 😄"],

    ["obrigada",
     "Disponha! 😄"],

    ["valeu",
     "Tamo junto!"],

    ["tchau",
     "Até mais! 👋"],

    ["ate mais",
     "Até a próxima!"],

    ["o que e rede neural",
     "Uma rede neural é um modelo computacional capaz de aprender padrões a partir de dados."],

    ["o que e aprendizado de maquina",
     "Aprendizado de máquina é uma área da inteligência artificial em que modelos aprendem padrões a partir de dados."],

    ["quem criou voce",
     "Eu sou um projeto de inteligência artificial desenvolvido em Python."]
]


# ============================================================
# CORPUS
# ============================================================

class Corpus:

    def __init__(self):

        self.dados = []

        self.carregar()


    def carregar(self):

        if not os.path.exists(
            ARQ_CORPUS
        ):

            self.dados = [

                {
                    "entrada": x[0],
                    "saida": x[1]
                }

                for x in CORPUS_INICIAL
            ]

            self.salvar()

            return


        try:

            with open(
                ARQ_CORPUS,
                "r",
                encoding="utf-8"
            ) as f:

                self.dados = json.load(f)

        except Exception:

            self.dados = []

            self.salvar()


    def salvar(self):

        with open(
            ARQ_CORPUS,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.dados,
                f,
                ensure_ascii=False,
                indent=4
            )


    def adicionar(
        self,
        entrada,
        saida
    ):

        entrada = entrada.strip()
        saida = saida.strip()

        if not entrada or not saida:

            return False

        self.dados.append({

            "entrada": entrada,
            "saida": saida

        })

        self.salvar()

        return True


# ============================================================
# VOCABULÁRIO
# ============================================================

class Vocabulario:

    ESPECIAIS = [

        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"

    ]


    def __init__(
        self,
        corpus
    ):

        self.tokens = []
        self.token_id = {}
        self.id_token = {}

        self.construir(
            corpus
        )


    def construir(
        self,
        corpus
    ):

        conjunto = set(
            self.ESPECIAIS
        )

        for item in corpus.dados:

            conjunto.update(
                tokenizar(
                    item["entrada"]
                )
            )

            conjunto.update(
                tokenizar(
                    item["saida"]
                )
            )

        self.tokens = sorted(
            conjunto
        )

        self.token_id = {

            token: i

            for i, token
            in enumerate(
                self.tokens
            )

        }

        self.id_token = {

            i: token

            for i, token
            in enumerate(
                self.tokens
            )

        }


    def salvar(self):

        with open(
            ARQ_VOCABULARIO,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.tokens,
                f,
                ensure_ascii=False,
                indent=2
            )


    def codificar(
        self,
        texto
    ):

        unk = self.token_id[
            "<UNK>"
        ]

        return [

            self.token_id.get(
                token,
                unk
            )

            for token
            in tokenizar(texto)

        ]


    def decodificar(
        self,
        ids
    ):

        tokens = []

        for i in ids:

            token = self.id_token.get(
                int(i),
                "<UNK>"
            )

            if token in (
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ):

                continue

            tokens.append(
                token
            )

        texto = ""

        for token in tokens:

            if token in (
                ".",
                ",",
                "!",
                "?",
                ":",
                ";"
            ):

                texto = texto.rstrip()
                texto += token

            else:

                if texto:

                    texto += " "

                texto += token

        return texto


# ============================================================
# FUNÇÕES
# ============================================================

def softmax(x):

    x = x - np.max(x)

    exp = np.exp(x)

    return exp / (
        np.sum(exp) + 1e-9
    )


def sigmoid(x):

    return 1.0 / (
        1.0 + np.exp(-x)
    )


# ============================================================
# MODELO NEURAL
# ============================================================

class NucleoNeural:

    def __init__(
        self,
        vocabulario
    ):

        self.vocab = vocabulario

        self.vocab_size = len(
            self.vocab.tokens
        )

        self.dim = int(
            CONFIG["dimensao"]
        )

        self.oculta = int(
            CONFIG["oculta"]
        )

        self.inicializar()


    def inicializar(self):

        escala = 0.08

        self.embedding = (

            np.random.randn(
                self.vocab_size,
                self.dim
            ) * escala

        )

        # projeção da entrada

        self.Wq = (

            np.random.randn(
                self.dim,
                self.dim
            ) * escala

        )

        self.Wk = (

            np.random.randn(
                self.dim,
                self.dim
            ) * escala

        )

        self.Wv = (

            np.random.randn(
                self.dim,
                self.dim
            ) * escala

        )

        # camada recorrente

        self.Wh = (

            np.random.randn(
                self.dim,
                self.oculta
            ) * escala

        )

        self.Uh = (

            np.random.randn(
                self.oculta,
                self.oculta
            ) * escala

        )

        self.bh = np.zeros(
            self.oculta
        )

        # saída

        self.Wo = (

            np.random.randn(
                self.oculta,
                self.vocab_size
            ) * escala

        )

        self.bo = np.zeros(
            self.vocab_size
        )


    # ========================================================
    # ATENÇÃO
    # ========================================================

    def atencao(
        self,
        estados
    ):

        if len(estados) == 0:

            return np.zeros(
                self.dim
            )

        X = np.array(
            estados
        )

        Q = np.dot(
            X[-1],
            self.Wq
        )

        K = np.dot(
            X,
            self.Wk
        )

        V = np.dot(
            X,
            self.Wv
        )

        scores = np.dot(
            K,
            Q
        ) / np.sqrt(
            self.dim
        )

        pesos = softmax(
            scores
        )

        contexto = np.sum(
            V * pesos[:, None],
            axis=0
        )

        return contexto


    # ========================================================
    # ESTADO
    # ========================================================

    def processar_contexto(
        self,
        ids
    ):

        h = np.zeros(
            self.oculta
        )

        estados = []

        for token_id in ids:

            x = self.embedding[
                token_id
            ]

            h = np.tanh(

                np.dot(
                    x,
                    self.Wh
                )

                +

                np.dot(
                    h,
                    self.Uh
                )

                +

                self.bh

            )

            estados.append(
                x.copy()
            )

        contexto = self.atencao(
            estados
        )

        return h, contexto


    # ========================================================
    # PREDIÇÃO
    # ========================================================

    def prever(
        self,
        h,
        contexto
    ):

        # Ajusta dimensão do contexto
        if len(contexto) != self.oculta:

            contexto_expandido = np.resize(
                contexto,
                self.oculta
            )

        else:

            contexto_expandido = contexto


        h_final = (

            h
            +
            contexto_expandido

        )


        logits = (

            np.dot(
                h_final,
                self.Wo
            )

            +

            self.bo

        )

        return softmax(
            logits
        )


    # ========================================================
    # SALVAR
    # ========================================================

    def salvar(self):

        np.savez_compressed(

            ARQ_MODELO,

            embedding=self.embedding,

            Wq=self.Wq,

            Wk=self.Wk,

            Wv=self.Wv,

            Wh=self.Wh,

            Uh=self.Uh,

            bh=self.bh,

            Wo=self.Wo,

            bo=self.bo

        )


    # ========================================================
    # CARREGAR
    # ========================================================

    def carregar(self):

        if not os.path.exists(
            ARQ_MODELO
        ):

            return False

        try:

            dados = np.load(
                ARQ_MODELO
            )

            self.embedding = (
                dados["embedding"]
            )

            self.Wq = (
                dados["Wq"]
            )

            self.Wk = (
                dados["Wk"]
            )

            self.Wv = (
                dados["Wv"]
            )

            self.Wh = (
                dados["Wh"]
            )

            self.Uh = (
                dados["Uh"]
            )

            self.bh = (
                dados["bh"]
            )

            self.Wo = (
                dados["Wo"]
            )

            self.bo = (
                dados["bo"]
            )

            return True

        except Exception:

            return False


# ============================================================
# GERADOR
# ============================================================

class Gerador:

    def __init__(
        self,
        modelo
    ):

        self.modelo = modelo


    def escolher(
        self,
        probabilidades
    ):

        temperatura = max(
            0.1,
            float(
                CONFIG[
                    "temperatura"
                ]
            )
        )

        logits = np.log(
            probabilidades + 1e-9
        )

        logits /= temperatura

        probabilidades = softmax(
            logits
        )

        return int(
            np.random.choice(
                len(probabilidades),
                p=probabilidades
            )
        )


    def gerar(
        self,
        entrada
    ):

        ids = (
            self.modelo
            .vocab
            .codificar(
                entrada
            )
        )

        if not ids:

            return ""


        limite_contexto = int(
            CONFIG["contexto"]
        )

        ids = ids[
            -limite_contexto:
        ]


        h, contexto = (
            self.modelo
            .processar_contexto(
                ids
            )
        )


        bos = (
            self.modelo
            .vocab
            .token_id[
                "<BOS>"
            ]
        )

        eos = (
            self.modelo
            .vocab
            .token_id[
                "<EOS>"
            ]
        )


        token_atual = bos

        resposta = []


        for _ in range(
            int(
                CONFIG[
                    "max_tokens"
                ]
            )
        ):

            x = (
                self.modelo
                .embedding[
                    token_atual
                ]
            )


            h = np.tanh(

                np.dot(
                    x,
                    self.modelo.Wh
                )

                +

                np.dot(
                    h,
                    self.modelo.Uh
                )

                +

                self.modelo.bh

            )


            probabilidades = (
                self.modelo.prever(
                    h,
                    contexto
                )
            )


            # Não gerar tokens especiais
            probabilidades[
                self.modelo.vocab.token_id[
                    "<PAD>"
                ]
            ] = 0


            probabilidades[
                self.modelo.vocab.token_id[
                    "<UNK>"
                ]
            ] *= 0.05


            soma = np.sum(
                probabilidades
            )


            if soma <= 0:

                break


            probabilidades /= soma


            token_atual = (
                self.escolher(
                    probabilidades
                )
            )


            if token_atual == eos:

                break


            resposta.append(
                token_atual
            )


        return (
            self.modelo
            .vocab
            .decodificar(
                resposta
            )
        )


# ============================================================
# MEMÓRIA
# ============================================================

class Memoria:

    def __init__(self):

        self.dados = []

        self.carregar()


    def carregar(self):

        if not os.path.exists(
            ARQ_MEMORIA
        ):

            return

        try:

            with open(
                ARQ_MEMORIA,
                "r",
                encoding="utf-8"
            ) as f:

                self.dados = json.load(
                    f
                )

        except Exception:

            self.dados = []


    def salvar(self):

        with open(
            ARQ_MEMORIA,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.dados,
                f,
                ensure_ascii=False,
                indent=4
            )


    def adicionar(
        self,
        autor,
        texto
    ):

        self.dados.append({

            "autor": autor,

            "texto": texto,

            "data": datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S"
            )

        })

        self.salvar()


    def ultimas(
        self,
        quantidade=6
    ):

        return self.dados[
            -quantidade:
        ]


    def limpar(self):

        self.dados = []

        self.salvar()


# ============================================================
# NEXA
# ============================================================

class Nexa:

    def __init__(self):

        print(
            "Inicializando NEXA..."
        )

        self.corpus = Corpus()

        self.vocab = Vocabulario(
            self.corpus
        )

        self.vocab.salvar()

        self.memoria = Memoria()

        self.modelo = NucleoNeural(
            self.vocab
        )

        carregado = (
            self.modelo.carregar()
        )

        if carregado:

            print(
                "Modelo neural carregado."
            )

        else:

            print(
                "Modelo novo criado."
            )

        self.gerador = Gerador(
            self.modelo
        )


    # ========================================================
    # TREINAR
    # ========================================================

    def treinar(self):

        print()

        print(
            "Preparando treinamento..."
        )

        print(
            "Exemplos:",
            len(
                self.corpus.dados
            )
        )

        print(
            "Vocabulário:",
            len(
                self.vocab.tokens
            )
        )

        print()

        # ----------------------------------------------------
        # OBSERVAÇÃO:
        # Esta versão utiliza um treinamento simplificado.
        # O objetivo é criar a arquitetura local.
        # ----------------------------------------------------

        taxa = float(
            CONFIG["taxa"]
        )

        epocas = int(
            CONFIG["epocas"]
        )

        for epoca in range(
            epocas
        ):

            perda = 0.0

            quantidade = 0


            for item in self.corpus.dados:

                entrada = (
                    self.vocab
                    .codificar(
                        item["entrada"]
                    )
                )

                saida = (
                    self.vocab
                    .codificar(
                        item["saida"]
                    )
                )


                if not entrada or not saida:

                    continue


                h, contexto = (
                    self.modelo
                    .processar_contexto(
                        entrada
                    )
                )


                alvo = saida[0]


                probabilidades = (
                    self.modelo.prever(
                        h,
                        contexto
                    )
                )


                perda -= np.log(
                    max(
                        probabilidades[
                            alvo
                        ],
                        1e-9
                    )
                )


                erro = (
                    probabilidades.copy()
                )

                erro[
                    alvo
                ] -= 1


                # Gradiente simplificado
                # da camada de saída.

                if len(contexto) == self.modelo.oculta:

                    representacao = (
                        h + contexto
                    )

                else:

                    representacao = (
                        h
                        +
                        np.resize(
                            contexto,
                            self.modelo.oculta
                        )
                    )


                grad_Wo = np.outer(
                    representacao,
                    erro
                )

                grad_bo = erro


                norma = np.linalg.norm(
                    grad_Wo
                )


                limite = float(
                    CONFIG[
                        "grad_clip"
                    ]
                )


                if norma > limite:

                    grad_Wo *= (
                        limite
                        /
                        norma
                    )


                self.modelo.Wo -= (
                    taxa
                    *
                    grad_Wo
                )

                self.modelo.bo -= (
                    taxa
                    *
                    grad_bo
                )


                quantidade += 1


            media = (
                perda
                /
                max(
                    quantidade,
                    1
                )
            )


            if (
                epoca == 0
                or
                (epoca + 1) % 50 == 0
                or
                epoca == epocas - 1
            ):

                print(
                    f"Época "
                    f"{epoca + 1}/"
                    f"{epocas}"
                    f" | perda: "
                    f"{media:.4f}"
                )


        self.modelo.salvar()

        print()

        print(
            "NEXA: treinamento concluído."
        )

        print()


    # ========================================================
    # ENSINAR
    # ========================================================

    def ensinar(
        self,
        entrada,
        saida
    ):

        sucesso = (
            self.corpus.adicionar(
                entrada,
                saida
            )
        )


        if not sucesso:

            print(
                "NEXA: dados inválidos."
            )

            return


        # Atualiza o vocabulário.

        self.vocab = Vocabulario(
            self.corpus
        )

        self.vocab.salvar()


        print()

        print(
            "NEXA: novo conhecimento "
            "adicionado ao corpus."
        )

        print(
            "Execute /treinar para "
            "incorporá-lo ao modelo."
        )

        print()


    # ========================================================
    # RESPONDER
    # ========================================================

    def responder(
        self,
        mensagem
    ):

        historico = (
            self.memoria
            .ultimas(
                6
            )
        )


        contexto = ""


        for item in historico:

            contexto += (
                item["autor"]
                + ": "
                + item["texto"]
                + " "
            )


        contexto += (
            "Você: "
            + mensagem
        )


        resposta = (
            self.gerador
            .gerar(
                contexto
            )
        )


        if not resposta:

            resposta = (
                "Ainda estou aprendendo "
                "a responder isso."
            )


        self.memoria.adicionar(
            "Você",
            mensagem
        )

        self.memoria.adicionar(
            "NEXA",
            resposta
        )


        return resposta


# ============================================================
# INTERFACE
# ============================================================

def banner():

    print()

    print(
        "=" * 66
    )

    print(
        "                         NEXA AI"
    )

    print(
        "                          v0.9"
    )

    print(
        "=" * 66
    )

    print(
        "Núcleo neural local"
    )

    print(
        "Mecanismo de atenção"
    )

    print(
        "Memória persistente"
    )

    print(
        "Corpus treinável"
    )

    print(
        "Sem API externa"
    )

    print(
        "=" * 66
    )

    print()


def ajuda():

    print()

    print(
        "COMANDOS DISPONÍVEIS"
    )

    print()

    print(
        "/ajuda"
    )

    print(
        "/status"
    )

    print(
        "/treinar"
    )

    print(
        "/ensinar"
    )

    print(
        "/memoria"
    )

    print(
        "/limpar"
    )

    print(
        "/temperatura"
    )

    print(
        "/sair"
    )

    print()


def mostrar_status(
    nexa
):

    print()

    print(
        "=============================="
    )

    print(
        "       STATUS DA NEXA"
    )

    print(
        "=============================="
    )

    print(
        "Versão:",
        VERSAO
    )

    print(
        "Exemplos:",
        len(
            nexa.corpus.dados
        )
    )

    print(
        "Vocabulário:",
        len(
            nexa.vocab.tokens
        )
    )

    print(
        "Dimensão:",
        nexa.modelo.dim
    )

    print(
        "Camada oculta:",
        nexa.modelo.oculta
    )

    print(
        "Temperatura:",
        CONFIG[
            "temperatura"
        ]
    )

    print(
        "Memórias:",
        len(
            nexa.memoria.dados
        )
    )

    print(
        "API:",
        "DESATIVADA"
    )

    print()


def mostrar_memoria(
    nexa
):

    print()

    if not nexa.memoria.dados:

        print(
            "A memória está vazia."
        )

        return


    print(
        "===== MEMÓRIA ====="
    )


    for item in (
        nexa.memoria.dados
    ):

        print(

            f'[{item["data"]}] '
            f'{item["autor"]}: '
            f'{item["texto"]}'

        )


    print()


def ensinar(
    nexa
):

    print()

    print(
        "===== ENSINAR NEXA ====="
    )

    print()


    entrada = input(
        "Pergunta/entrada: "
    ).strip()


    saida = input(
        "Resposta esperada: "
    ).strip()


    if not entrada or not saida:

        print(
            "Preencha todos os campos."
        )

        return


    nexa.ensinar(
        entrada,
        saida
    )


def mudar_temperatura():

    try:

        valor = float(
            input(
                "Temperatura "
                "(0.1 até 2.0): "
            )
        )


        valor = max(
            0.1,
            min(
                2.0,
                valor
            )
        )


        CONFIG[
            "temperatura"
        ] = valor


        salvar_config(
            CONFIG
        )


        print(
            "Temperatura:",
            valor
        )


    except ValueError:

        print(
            "Valor inválido."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    nexa = Nexa()

    banner()


    while True:

        try:

            mensagem = input(
                "Você: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()

            print(
                "NEXA: Até mais! 👋"
            )

            break


        if not mensagem:

            continue


        comando = mensagem.lower()


        if comando == "/sair":

            print(
                "NEXA: Encerrando..."
            )

            break


        if comando == "/ajuda":

            ajuda()

            continue


        if comando == "/status":

            mostrar_status(
                nexa
            )

            continue


        if comando == "/treinar":

            nexa.treinar()

            continue


        if comando == "/ensinar":

            ensinar(
                nexa
            )

            continue


        if comando == "/memoria":

            mostrar_memoria(
                nexa
            )

            continue


        if comando == "/limpar":

            nexa.memoria.limpar()

            print(
                "NEXA: memória limpa."
            )

            continue


        if comando == "/temperatura":

            mudar_temperatura()

            continue


        resposta = (
            nexa.responder(
                mensagem
            )
        )


        print()

        print(
            "NEXA:",
            resposta
        )

        print()


# ============================================================
# INÍCIO
# ============================================================

if __name__ == "__main__":

    main()