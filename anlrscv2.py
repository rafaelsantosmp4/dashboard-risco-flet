import flet as ft
import yfinance as yf
import numpy as np
import requests
import asyncio
import re
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import traceback

# Configura o matplotlib para usar o backend 'agg', que não exige interface gráfica, ideal para gerar imagens em background.
matplotlib.use("agg")

# ==========================================
# Classes de Domínio
# ==========================================
class AtivoFinanceiro:
    """Classe base que representa um ativo financeiro genérico."""
    def __init__(self, ticker):
        self.ticker = ticker.upper() # O símbolo do ativo no mercado (ex: AAPL, PETR4.SA)
        self.nome = self.ticker
        self.dados_historicos = None # Armazenará o DataFrame com os preços
        self.preco_atual = 0.0
        self.retorno_ano = 0.0
        self.max_periodo = 0.0
        self.min_periodo = 0.0
        self.drawdown_maximo = 0.0 # Maior queda do preço a partir de um pico

    def buscar_dados(self, periodo="1y"):
        """Busca os dados históricos do ativo usando a biblioteca yfinance."""
        try:
            info = yf.Ticker(self.ticker).info
            self.nome = info.get('longName', info.get('shortName', self.ticker))
        except:
            pass # Se falhar ao buscar o nome, mantém o ticker como nome

        # Faz o download dos dados históricos no período especificado
        df = yf.download(self.ticker, period=periodo, progress=False)
        if df is None or df.empty or len(df) < 2:
            raise ValueError(f"Dados insuficientes para o ticker {self.ticker} no período {periodo}.")
        
        self.dados_historicos = df
        precos = df['Close'] # Pega apenas a coluna de preços de fechamento
        
        # Trata o caso onde yfinance retorna um DataFrame com MultiIndex nas colunas (ocorre em algumas versões)
        if hasattr(precos, 'iloc') and len(precos.shape) > 1:
            precos = precos.iloc[:, 0]
            
        if len(precos) == 0:
            raise ValueError("O ativo foi encontrado, mas não retornou preços para este período.")
            
        self.preco_atual  = float(precos.iloc[-1]) # Último preço da série
        preco_inicial     = float(precos.iloc[0])  # Primeiro preço da série
        
        # Calcula a variação percentual entre o primeiro e o último preço
        self.retorno_ano  = ((self.preco_atual - preco_inicial) / preco_inicial) * 100
        self.max_periodo  = float(df['High'].squeeze().max()) # Maior valor atingido no período
        self.min_periodo  = float(df['Low'].squeeze().min())  # Menor valor atingido no período
        
        # Calcula o drawdown (queda em relação ao pico histórico mais recente)
        picos = precos.cummax() # array com os valores máximos alcançados até cada ponto
        self.drawdown_maximo = float(((precos - picos) / picos).min() * 100)

    def calcular_risco(self):
        """Calcula a volatilidade anualizada baseada no desvio padrão dos retornos diários."""
        if self.dados_historicos is None or self.dados_historicos.empty:
            return 0.0
        precos = self.dados_historicos['Close']
        if len(precos.shape) > 1:
            precos = precos.iloc[:, 0]
        if len(precos) < 2:
            return 0.0
            
        # Calcula a mudança percentual diária
        retornos = precos.pct_change().dropna()
        if retornos.empty:
            return 0.0
            
        # Volatilidade anualizada = desvio padrão diário * raiz quadrada de 252 (dias úteis no ano)
        return float(np.std(retornos) * np.sqrt(252) * 100)

    def obter_dados_formatados(self):
        """Método que deve ser implementado nas subclasses para retornar dados em formato legível."""
        raise NotImplementedError

class AtivoNacional(AtivoFinanceiro):
    """Subclasse para ativos negociados em BRL (Reais)."""
    def obter_dados_formatados(self):
        return {
            "preco": f"R$ {self.preco_atual:.2f}", "preco_sec": "",
            "max":   f"R$ {self.max_periodo:.2f}", "max_sec":   "",
            "min":   f"R$ {self.min_periodo:.2f}", "min_sec":   "",
        }

class AtivoInternacional(AtivoFinanceiro):
    """Subclasse para ativos internacionais. Converte valores para BRL usando cotação do dólar."""
    def __init__(self, ticker):
        super().__init__(ticker)
        self.cotacao_dolar = 1.0

    def buscar_dados(self, periodo="1y"):
        super().buscar_dados(periodo)
        # Busca a cotação do dólar em relação ao Real (BRL=X)
        df_dolar = yf.download("BRL=X", period="1d", progress=False)
        if not df_dolar.empty:
            v = df_dolar['Close'].squeeze()
            self.cotacao_dolar = float(v) if isinstance(v, (np.float64, float)) else float(v.iloc[-1])

    def obter_dados_formatados(self):
        c = self.cotacao_dolar
        return {
            "preco":     f"$ {self.preco_atual:.2f}",
            "preco_sec": f"R$ {self.preco_atual * c:.2f}", # Valor em Reais (secundário)
            "max":       f"$ {self.max_periodo:.2f}",
            "max_sec":   f"R$ {self.max_periodo * c:.2f}",
            "min":       f"$ {self.min_periodo:.2f}",
            "min_sec":   f"R$ {self.min_periodo * c:.2f}",
        }

def criar_ativo(ticker):
    """Factory method: Decide qual classe de ativo instanciar com base no sufixo do ticker."""
    t = ticker.upper()
    # Ativos com final .SA (bolsa brasileira) ou com BRL no nome são nacionais
    return AtivoNacional(ticker) if (t.endswith(".SA") or t.endswith("BRL")) else AtivoInternacional(ticker)

def _normalizar_serie(valor):
    """Função utilitária para garantir que recebemos um pd.Series unidimensional (1D)."""
    if valor is None:
        return None
    if hasattr(valor, "squeeze"):
        valor = valor.squeeze() # Remove dimensões de tamanho 1
    if hasattr(valor, "ndim") and getattr(valor, "ndim", 1) > 1 and hasattr(valor, "iloc"):
        valor = valor.iloc[:, 0] # Pega a primeira coluna se for um DataFrame
    return valor

def gerar_grafico_b64(
    fonte,
    ticker=None,
    mostrar_mm_rapida=False,
    mostrar_mm_lenta=True,
    mostrar_bollinger=False,
    mostrar_volume_financeiro=False,
    mostrar_rsi=False,
):
    """Gera o gráfico usando matplotlib e o converte para uma string Base64 para exibir na interface Flet."""
    if hasattr(fonte, "dados_historicos"):
        df = fonte.dados_historicos
        ticker = ticker or getattr(fonte, "ticker", "ATIVO")
    else:
        df = fonte
        ticker = ticker or "ATIVO"

    if df is None or len(df) < 2:
        raise ValueError("Não há dados históricos suficientes para gerar o gráfico.")

    close = _normalizar_serie(df["Close"])
    if close is None or len(close) < 2:
        raise ValueError("Não foi possível acessar a série de preços de fechamento.")

    close = close.astype(float)

    rsi = None
    tem_dados_suficientes_rsi = len(close) > 14
    
    # Cálculo do RSI (Índice de Força Relativa) - indicador de momento para medir a magnitude e a velocidade das variações de preço
    if mostrar_rsi and tem_dados_suficientes_rsi:
        delta = close.diff() # Diferença de preço dia a dia

        ganho = delta.where(delta > 0, 0) # Apenas dias de alta
        perda = -delta.where(delta < 0, 0) # Apenas dias de baixa (positivo)

        # Média móvel dos ganhos e perdas em 14 períodos
        media_ganho = ganho.rolling(14).mean()
        media_perda = perda.rolling(14).mean()

        rs = media_ganho / media_perda # Força Relativa
        rsi = 100 - (100 / (1 + rs))   # RSI padronizado entre 0 e 100
    else:
        mostrar_rsi = False

    # Montante total de dinheiro (preço × quantidade) movimentado na compra e venda
    volume = None
    if "Volume" in df.columns:
        volume = _normalizar_serie(df["Volume"])
        if volume is not None:
            volume = volume.astype(float)

        tem_volume = bool(mostrar_volume_financeiro and volume is not None and len(volume) == len(close))

    axv = None
    axrsi = None

    # Configuração dos subplots dependendo de quais indicadores estão ativos
    if mostrar_rsi and tem_volume:
        # Gráfico triplo: Preços, RSI e Volume
        fig, (ax, axrsi, axv) = plt.subplots(
            3, 1, figsize=(12, 9),
            gridspec_kw={"height_ratios": [4, 1, 1]}, sharex=True
        )
    elif mostrar_rsi:
        # Gráfico duplo: Preços e RSI
        fig, (ax, axrsi) = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
    elif tem_volume:
        # Gráfico duplo: Preços e Volume
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
    else:
        # Gráfico simples: Apenas Preços
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plota a linha principal de preços
    ax.plot(close.index, close, color='#3282F6', linewidth=2, label="Preço de Fechamento")
    
    # Desenha as Velas (Candlesticks)
    open_p = _normalizar_serie(df["Open"]).astype(float)
    high_p = _normalizar_serie(df["High"]).astype(float)
    low_p = _normalizar_serie(df["Low"]).astype(float)
    cores = ['#52A56A' if c >= o else '#D94B43' for c, o in zip(close, open_p)]
    ax.vlines(close.index, ymin=low_p, ymax=high_p, color=cores, linewidth=1, zorder=3)
    alturas = np.abs(close - open_p)
    bases = np.minimum(open_p, close)
    ax.bar(close.index, height=alturas, bottom=bases, color=cores, edgecolor=cores, linewidth=0.5, width=0.6, zorder=4)


    # Média Móvel de 20 períodos
    if mostrar_mm_rapida and len(close) >= 20:
        mm_rapida = close.rolling(20).mean()
        ax.plot(
            close.index, mm_rapida,
            color='#00C2FF', linewidth=1.4, linestyle='--',
            label="Média Móvel (20 dias)"
        )

    # Média Móvel de 50 períodos
    if mostrar_mm_lenta and len(close) >= 50:
        mm_lenta = close.rolling(50).mean()
        ax.plot(
            close.index, mm_lenta,
            color='#FFA500', linewidth=1.6, linestyle='--',
            label="Média Móvel (50 dias)"
        )

    # Bandas de Bollinger: medem a volatilidade adicionando/subtraindo 2 desvios padrões
    if mostrar_bollinger and len(close) >= 20:
        media = close.rolling(20).mean()
        desvio = close.rolling(20).std()
        banda_superior = media + (2 * desvio)
        banda_inferior = media - (2 * desvio)

        ax.plot(close.index, banda_superior, color='#3CB371', linewidth=0.8, alpha=0.8, label="Banda Superior")
        ax.plot(close.index, banda_inferior, color='#3CB371', linewidth=0.8, alpha=0.8, label="Banda Inferior")
        ax.plot(close.index, media, color='#3CB371', linewidth=0.5, linestyle='--', alpha=0.5) # Linha central opcional

        # Lógica de Sinais (Setas)
        # Identifica os dias onde o preço cruzou as bandas
        sinal_compra = close <= banda_inferior
        sinal_venda = close >= banda_superior

        # Calcula um pequeno recuo (offset) visual para a seta não ficar colada na linha
        offset = close.std() * 0.2

        # Plota setas verdes (^) para sinais de compra
        ax.scatter(
            close.index[sinal_compra], 
            close[sinal_compra] - offset, 
            marker='^', color='lime', s=60, zorder=5, label="Sinal de Compra"
        )
        
        # Plota setas vermelhas (v) para sinais de venda
        ax.scatter(
            close.index[sinal_venda], 
            close[sinal_venda] + offset, 
            marker='v', color='red', s=60, zorder=5, label="Sinal de Venda"
        )

    # Plota o Volume Financeiro (Preço * Volume Negociado)
    if tem_volume and axv is not None:
        volume_financeiro = close * volume
        axv.fill_between(volume_financeiro.index, 0, volume_financeiro, color='#CCCCCC', alpha=0.3)
        axv.set_facecolor('#1E1E1E') 
        axv.patch.set_facecolor('#1E1E1E')
        axv.set_ylabel("Volume financeiro", color='#CCCCCC', fontsize=10, labelpad=8)
        axv.tick_params(colors='white', labelsize=8)
        for sp in axv.spines.values():
            sp.set_color('#444444')
        axv.grid(color='#333333', linestyle='--', linewidth=0.5)
        handles_v, labels_v = axv.get_legend_handles_labels()
        if handles_v:
            axv.legend(facecolor='#1E1E1E', edgecolor='#444444', labelcolor='white', loc="upper left")

    # Configurações estéticas gerais do gráfico de preços
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.set_title(
        f"Evolução Histórica e Tendência - {ticker}",
        color='white', fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel("Data", color='#CCCCCC', fontsize=10, labelpad=10)
    ax.set_ylabel("Valor Cotado", color='#CCCCCC', fontsize=10, labelpad=10)
    ax.tick_params(colors='white', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#444444')
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(facecolor='#1E1E1E', edgecolor='#444444', labelcolor='white', loc="upper left")

    # Plota o RSI
    if mostrar_rsi and rsi is not None:
        axrsi.plot(rsi.index, rsi, color="#9B59B6", linewidth=1.5, label="RSI (14)")
        axrsi.axhline(70, color="red", linestyle="--", linewidth=1) # Linha de sobrecompra
        axrsi.axhline(30, color="green", linestyle="--", linewidth=1) # Linha de sobrevenda
        axrsi.set_ylim(0,100)
        axrsi.set_facecolor("#1E1E1E")
        axrsi.set_ylabel("RSI", color="white", fontsize=9)
        axrsi.set_yticks([30, 50, 70])
        axrsi.tick_params(colors="white", labelsize=8)
        axrsi.grid(color="#333333", linestyle="--", linewidth=0.5)
        handles, labels = axrsi.get_legend_handles_labels()
        if handles:
            axrsi.legend(facecolor="#1E1E1E", edgecolor="#444444", labelcolor="white")

    fig.tight_layout() # Ajusta o espaçamento para não cortar textos
    
    # Salva o gráfico em um buffer de memória ao invés de um arquivo no disco
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
    buf.seek(0) # Volta o cursor do buffer para o início
    
    # Converte os bytes da imagem para uma string Base64 e fecha a figura
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# ==========================================
# UI (Interface Gráfica com Flet)
# ==========================================
def main(page: ft.Page):
    """Função principal que constrói e gerencia a interface gráfica."""
    page.title = "Dashboard Analítico de Risco"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1500
    page.window.height = 900
    page.padding = ft.Padding.only(top=50, bottom=40, left=0, right=0)
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = "auto"
    
    def on_keyboard(e):
        if e.key == "F11":
            page.window.full_screen = not page.window.full_screen
            page.update()
            
        elif e.key == "Escape":
            page.window.full_screen = False
            page.update()
            
    page.on_keyboard_event = on_keyboard

    # ── estado ────────────────────────────────────────────────────────────────
    estado_app          = {"ticker": None, "periodo": "1y"}
    cache_ativos: dict  = {}          # Guarda dados cacheados por "TICKER|periodo" para evitar reprocessamento
    historico_tickers   = []          # Lista dos últimos tickers buscados
    refs_animacao       = []          # Referências aos Textos que terão números animados
    ref_grafico_img     = ft.Ref[ft.Image]()
    linha_indicadores   = None

    estado_indicadores = {
        "mm_rapida": True,
        "mm_lenta": True,
        "bollinger": True,
        "volume_financeiro": True,
    }

    def _chave_atual():
        """Retorna uma string única para identificar os dados no cache."""
        ticker = estado_app.get("ticker")
        periodo = estado_app.get("periodo")
        if not ticker or not periodo:
            return None
        return f"{ticker.upper()}|{periodo}"

    async def atualizar_grafico_indicadores():
        """Refaz o gráfico quando um switch de indicador é alternado."""
        chave = _chave_atual()
        if not chave or chave not in cache_ativos:
            return

        dados = cache_ativos[chave]
        df = dados.get("df")
        if df is None:
            return

        # Gera o novo gráfico base64 em uma thread separada para não travar a interface
        b64 = await asyncio.to_thread(
            gerar_grafico_b64,
            df,
            dados.get("ticker"),
            switch_mm_rapida.value,
            switch_mm_lenta.value,
            switch_bollinger.value,
            switch_volume_financeiro.value,
            switch_rsi.value,
        )

        dados["grafico_b64"] = b64
        if ref_grafico_img.current:
            ref_grafico_img.current.src = f"data:image/png;base64,{b64}"
        page.update()

    def _on_toggle_indicador(e):
        # Evento disparado quando um switch muda de valor
        asyncio.ensure_future(atualizar_grafico_indicadores())

    # ── tema ──────────────────────────────────────────────────────────────────
    def alternar_tema(e):
        """Alterna entre os modos Claro e Escuro da interface."""
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
            botao_tema.icon    = ft.Icons.DARK_MODE_OUTLINED
            botao_tema.tooltip = "Mudar para Modo Escuro"
        else:
            page.theme_mode = ft.ThemeMode.DARK
            botao_tema.icon    = ft.Icons.LIGHT_MODE_OUTLINED
            botao_tema.tooltip = "Mudar para Modo Claro"
        page.update()

    botao_tema = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE_OUTLINED, tooltip="Mudar para Modo Claro", on_click=alternar_tema,
        mouse_cursor=ft.MouseCursor.CLICK, margin=ft.Margin.only(right=20)
    )
    titulo = ft.Text("Dashboard Analítico de Risco", size=28, weight=ft.FontWeight.BOLD)
    painel_dados = ft.Column(
        visible=False, spacing=20,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER, width=1000,
    )

    # ── histórico ─────────────────────────────────────────────────────────────
    linha_historico = ft.Row(controls=[], wrap=True, spacing=8, run_spacing=8, visible=False)

    def atualizar_historico(ticker):
        """Mantém uma lista visual dos últimos 8 tickers pesquisados."""
        ticker = ticker.upper()
        if ticker in historico_tickers:
            historico_tickers.remove(ticker) # Traz para o início se já existir
        historico_tickers.insert(0, ticker)
        del historico_tickers[8:] # Mantém no máximo 8 itens
        linha_historico.controls.clear()
        linha_historico.controls.append(
            ft.Text("Recentes:", size=11, color=ft.Colors.GREY_500, weight=ft.FontWeight.BOLD)
        )
            
        for t in historico_tickers:
            tk = t
            linha_historico.controls.append(
                ft.Container(
                    content=ft.Text(tk, size=12, weight=ft.FontWeight.W_600, color=ft.Colors.GREY_300),
                    bgcolor="#2a2b2e",
                    border_radius=20,
                    padding=ft.Padding(12, 6, 12, 6),
                    border=ft.Border(
                        ft.BorderSide(1, "#3a3b3e"), ft.BorderSide(1, "#3a3b3e"),
                        ft.BorderSide(1, "#3a3b3e"), ft.BorderSide(1, "#3a3b3e"),
                    ),
                    on_click=lambda e, t=tk: asyncio.ensure_future(executar_calculo(novo_ticker=t)), # Refaz a busca ao clicar
                    ink=True,
                )
            )
        linha_historico.visible = True

    # ── skeleton ──────────────────────────────────────────────────────────────
    # Elementos visuais (Skeletons) exibidos enquanto os dados estão sendo carregados (efeito de "loading").
    _sk_refs: list[ft.Container] = []

    def _bloco(w, h):
        """Cria um bloco retangular animado para o Skeleton."""
        c = ft.Container(
            bgcolor="#3a3b3e", border_radius=6, width=w, height=h,
            opacity=1.0,
            animate_opacity=ft.Animation(600, ft.AnimationCurve.EASE_IN_OUT),
        )
        _sk_refs.append(c)
        return c

    def _card_sk():
        """Cria um Skeleton com formato de Card."""
        return ft.Container(
            content=ft.Column(
                [_bloco(60,11), ft.Container(height=6),
                 _bloco(80,20), ft.Container(height=4), _bloco(45,10)],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            ),
            bgcolor="#2a2b2e", padding=15, border_radius=10, height=110, expand=True,
        )

    sk_row1 = ft.Row([_card_sk() for _ in range(4)],
                     alignment=ft.MainAxisAlignment.SPACE_BETWEEN, visible=False, width=1000)
    sk_row2 = ft.Row([_card_sk() for _ in range(2)], visible=False, width=1000)

    sk_grafico = ft.Container(
        bgcolor="#2a2b2e", border_radius=10, height=300, width=1000,
        visible=False, padding=20,
        content=ft.Column(
            [_bloco(260, 14), ft.Container(height=16), _bloco(940, 200)],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
        ),
    )

    _pulsando = False

    async def _pulsar():
        """Anima a opacidade dos blocos do Skeleton para criar um efeito de pulsação (loading)."""
        nonlocal _pulsando
        _pulsando = True
        while sk_row1.visible:
            for b in _sk_refs:
                b.opacity = 0.25
            page.update()
            await asyncio.sleep(0.6)
            if not sk_row1.visible:
                break
            for b in _sk_refs:
                b.opacity = 1.0
            page.update()
            await asyncio.sleep(0.6)
        _pulsando = False

    def mostrar_skeletons():
        for b in _sk_refs:
            b.opacity = 1.0
        sk_row1.visible    = True
        sk_row2.visible    = True
        sk_grafico.visible = True
        page.update()
        asyncio.ensure_future(_pulsar())

    def esconder_skeletons():
        sk_row1.visible    = False
        sk_row2.visible    = False
        sk_grafico.visible = False

    switch_mm_rapida = ft.Switch(
        label="MM Rápida (20)",
        value=estado_indicadores["mm_rapida"],
        on_change=_on_toggle_indicador,
    )
    switch_mm_lenta = ft.Switch(
        label="MM Lenta (50)",
        value=estado_indicadores["mm_lenta"],
        on_change=_on_toggle_indicador,
    )
    switch_bollinger = ft.Switch(
        label="Bandas de Bollinger",
        value=estado_indicadores["bollinger"],
        on_change=_on_toggle_indicador,
    )
    switch_rsi = ft.Switch(
        label="RSI (14)",
        value=True,
        on_change=_on_toggle_indicador,
    )
    switch_volume_financeiro = ft.Switch(
        label="Volume financeiro",
        value=estado_indicadores["volume_financeiro"],
        on_change=_on_toggle_indicador,
    )

    linha_indicadores = ft.Container(
        padding=ft.Padding.symmetric(vertical=4),
        content=ft.Row(
            controls=[
                switch_mm_rapida,
                switch_mm_lenta,
                switch_bollinger,
                switch_rsi,
                switch_volume_financeiro,
            ],
            wrap=True,
            spacing=18,
            run_spacing=8,
        ),
    )

    # ── cards com contador animado ────────────────────────────────────────────
    def criar_card_info(titulo_card, valor, subtitulo="", cor_valor=ft.Colors.WHITE, valor_secundario="", primeira_vez=False):
        """Cria um Card visual que armazenará uma informação financeira (ex: Preço, Risco)."""
        # Referências (ft.Ref) para poder atualizar o texto do card posteriormente (durante a animação)
        ref_p = ft.Ref[ft.Text]()
        ref_s = ft.Ref[ft.Text]() if valor_secundario else None
        els = [ft.Text(titulo_card, size=11, color=ft.Colors.GREY_400, weight=ft.FontWeight.BOLD)]
        
        # Inicializa o valor com um traço ("—"), será atualizado via animação
        if valor_secundario:
            els.append(ft.Row(
                [ft.Text("—", size=22, color=cor_valor, weight=ft.FontWeight.BOLD, ref=ref_p),
                 ft.Text("",  size=12, color=ft.Colors.GREY_500, ref=ref_s)],
                alignment=ft.MainAxisAlignment.CENTER, spacing=6,
            ))
        else:
            els.append(ft.Text("—", size=22, color=cor_valor, weight=ft.FontWeight.BOLD, ref=ref_p))
            
        if subtitulo:
            els.append(ft.Text(subtitulo, size=12, color=ft.Colors.GREY_500))
            
        # Registra esse card na lista de animação para ser processado depois
        refs_animacao.append({
            "ref": ref_p, "ref_sec": ref_s,
            "valor_final": valor,
            "valor_sec_final": f"({valor_secundario})" if valor_secundario else "",
        })
        
        return ft.Container(
            content=ft.Column(els, alignment=ft.MainAxisAlignment.CENTER,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor="#2a2b2e", padding=15, border_radius=10, height=110, expand=True,
            
            # Animações de entrada: surgem de baixo (offset) para cima, mudando a opacidade
            opacity=0 if primeira_vez else 1,
            offset=ft.Offset(0, 0.3) if primeira_vez else ft.Offset(0, 0),
            
            animate_opacity=ft.Animation(600, ft.AnimationCurve.DECELERATE),
            animate_offset=ft.Animation(600, ft.AnimationCurve.DECELERATE),
        )

    async def animar_contadores():
        """Faz a animação dos números crescendo gradualmente de 0 até o valor final."""
        def extrair(s):
            # Extrai apenas os números e formatação decimal de uma string (ex: "R$ 15,50" -> 15.5)
            m = re.search(r"[-+]?\d[\d.,]*", s)
            if not m: return None
            try: return float(m.group().replace(".", "").replace(",", "."))
            except: return None

        def reconstruir(tmpl, n):
            # Injeta o número animado (n) de volta na string formatada original (tmpl)
            m = re.search(r"[-+]?\d[\d.,]*", tmpl)
            if not m: return tmpl
            s, e = m.span()
            casas = 2 if ("." in tmpl[s:e] or "," in tmpl[s:e]) else 1
            sinal = "+" if tmpl.startswith("+") else ""
            return tmpl[:s] + f"{sinal}{n:.{casas}f}" + tmpl[e:]

        FRAMES = 15
        DURACAO = 0.1

        async def animar(ref, vs, nf, ref_sec=None, vs_sec=None):
            for i in range(1, FRAMES + 1):
                # Função de interpolação curva para desacelerar perto do fim
                p = 1 - (1 - i / FRAMES) ** 2
                if ref.current:
                    ref.current.value = reconstruir(vs, nf * p)
                if ref_sec and ref_sec.current and vs_sec:
                    ns = extrair(vs_sec)
                    if ns is not None:
                        ref_sec.current.value = "(" + reconstruir(vs_sec[1:-1], ns * p) + ")"
                page.update()
                await asyncio.sleep(DURACAO / FRAMES)
            if ref.current: ref.current.value = vs
            if ref_sec and ref_sec.current and vs_sec: ref_sec.current.value = vs_sec
            page.update()

        for entrada in refs_animacao:
            ref = entrada["ref"]
            vs  = entrada["valor_final"]
            n   = extrair(vs)
            if n is None:
                if ref.current: ref.current.value = vs
                continue
            await animar(ref, vs, n,
                         ref_sec=entrada.get("ref_sec"),
                         vs_sec=entrada.get("valor_sec_final") or None)

    # ── montar painel a partir do cache ───────────────────────────────────────
    def _montar_painel(d, primeira_vez=False):
        """Constrói visualmente o painel preenchendo-o com os dados calculados do ativo."""
        st = d["strings"]
        refs_animacao.clear()
        
        # Primeira linha de cards (Preço, Retorno, Volatilidade, Drawdown)
        lc = ft.Row([
            criar_card_info("PREÇO ATUAL",       st["preco"],    valor_secundario=st["preco_sec"], primeira_vez=primeira_vez),
            criar_card_info(d["titulo_retorno"], d["retorno"],   cor_valor=d["cor_retorno"], primeira_vez=primeira_vez),
            criar_card_info("VOLATILIDADE",      d["risco"],     subtitulo=f"Risco: {d['nivel_risco']}", primeira_vez=primeira_vez),
            criar_card_info("MAIOR QUEDA",       d["drawdown"],  subtitulo="Drawdown", cor_valor=ft.Colors.RED_300, primeira_vez=primeira_vez),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        
        # Segunda linha de cards (Máxima e Mínima)
        ls = ft.Row([
            criar_card_info("MÁXIMA (Período)", st["max"], valor_secundario=st["max_sec"], primeira_vez=primeira_vez),
            criar_card_info("MÍNIMA (Período)", st["min"], valor_secundario=st["min_sec"], primeira_vez=primeira_vez),
        ])
        
        # O Gráfico gerado em Base64
        graf = ft.Container(
            content=ft.Image(
                ref=ref_grafico_img,
                src=f"data:image/png;base64,{d['grafico_b64']}",
                fit="contain",
            ),
            padding=0, border_radius=10, bgcolor="#1E1E1E",
            
            # Animações de entrada
            opacity=0 if primeira_vez else 1,
            offset=ft.Offset(0, 0.3) if primeira_vez else ft.Offset(0, 0),
            animate_opacity=ft.Animation(600, ft.AnimationCurve.DECELERATE),
            animate_offset=ft.Animation(600, ft.AnimationCurve.DECELERATE),
        )
        
        painel_dados.controls = [
            ft.Text(f"Ativo: {d['nome']} ({d['ticker']})", size=22, weight=ft.FontWeight.BOLD),
            seletor_periodo,
            linha_indicadores,
            lc, ls, graf,
        ]

    # ── lógica principal ──────────────────────────────────────────────────────
    async def executar_calculo(novo_ticker=None, novo_periodo=None):
        """Fluxo principal: recebe um ticker, busca dados, processa, salva no cache e exibe."""
        if novo_ticker: estado_app["ticker"] = novo_ticker
        if novo_periodo: estado_app["periodo"] = novo_periodo
        ticker_escolhido  = estado_app["ticker"]
        periodo_escolhido = estado_app["periodo"]
        if not ticker_escolhido:
            return

        chave = f"{ticker_escolhido.upper()}|{periodo_escolhido}"

        # cache hit → Se os dados já existem no cache, carrega rapidamente sem recarregar da API
        if chave in cache_ativos:
            atualizar_historico(ticker_escolhido)
            _montar_painel(cache_ativos[chave], primeira_vez=False)
            painel_dados.visible = True
            page.update()
            await atualizar_grafico_indicadores()
            await animar_contadores()
            return

        # cache miss → mostra skeletons de loading, busca na API usando thread secundária
        painel_dados.visible = False
        mostrar_skeletons()
        await asyncio.sleep(0) # Permite o event loop atualizar a interface

        try:
            ativo = criar_ativo(ticker_escolhido)
            
            # Executa requisição web (yf.download) em background usando threads
            await asyncio.to_thread(ativo.buscar_dados, periodo_escolhido)

            risco        = ativo.calcular_risco()
            cor_retorno  = ft.Colors.GREEN_400 if ativo.retorno_ano >= 0 else ft.Colors.RED_400
            sinal        = "+" if ativo.retorno_ano >= 0 else ""
            nivel_risco  = "ALTO" if risco > 40 else "MÉDIO" if risco > 20 else "BAIXO"
            titulos_ret = {
                "5d":  "RETORNO (5D)",
                "1mo": "RETORNO (1M)",
                "3mo": "RETORNO (3M)",
                "6mo": "RETORNO (6M)",
                "1y":  "RETORNO (1A)",
                "max": "RETORNO (MAX)"
            }
            titulo_do_card = titulos_ret.get(periodo_escolhido, "RETORNO")

            # Gera a imagem do gráfico (tarefa custosa do Matplotlib, movida para thread)
            grafico_b64 = await asyncio.to_thread(
                gerar_grafico_b64,
                ativo,
                ativo.ticker,
                switch_mm_rapida.value,
                switch_mm_lenta.value,
                switch_bollinger.value,
                switch_volume_financeiro.value,
                switch_rsi.value,
            )

            # Empacota os resultados
            dados = {
                "nome":           ativo.nome,
                "ticker":         ativo.ticker,
                "strings":        ativo.obter_dados_formatados(),
                "titulo_retorno": titulos_ret[periodo_escolhido],
                "retorno":        f"{sinal}{ativo.retorno_ano:.2f}%",
                "cor_retorno":    cor_retorno,
                "risco":          f"{risco:.1f}%",
                "nivel_risco":    nivel_risco,
                "drawdown":       f"{ativo.drawdown_maximo:.1f}%",
                "grafico_b64":    grafico_b64,
                "df":             ativo.dados_historicos,
            }
            
            # Guarda em cache
            cache_ativos[chave] = dados
            atualizar_historico(ticker_escolhido)
            
            # Prepara o painel com os cards transparentes (estado inicial da animação)
            _montar_painel(dados, primeira_vez=True)
            esconder_skeletons()
            painel_dados.visible = True
            page.update()

            # Dá um tempo mínimo para a interface processar o estado inicial
            await asyncio.sleep(0.05)

            # Dispara a animação dos cards alterando a opacidade e o offset
            for linha_cards in [painel_dados.controls[3], painel_dados.controls[4]]:
                for card in linha_cards.controls:
                    card.opacity = 1
                    card.offset = ft.Offset(0, 0)
            
            # Dispara a animação do gráfico
            grafico_container = painel_dados.controls[5]
            grafico_container.opacity = 1
            grafico_container.offset = ft.Offset(0, 0)
            
            page.update()
            
            # Anima os números paralelamente
            await animar_contadores()

        except Exception:
            # Em caso de falha (ticker não existe, sem internet, etc.), oculta loading e mostra erro
            erro = traceback.format_exc()
            esconder_skeletons()
            painel_dados.controls = [
                ft.Text("❌ Erro ao buscar dados:", color=ft.Colors.RED, weight=ft.FontWeight.BOLD),
                ft.Text(erro, color=ft.Colors.RED_200, size=12, selectable=True),
            ]
            painel_dados.visible = True
            page.update()

    # ── seletor de período ────────────────────────────────────────────────────
    # Botões na interface para trocar o intervalo de tempo analisado
    seletor_periodo = ft.SegmentedButton(
        selected=["1y"],
        segments=[
            ft.Segment(value="5d",  label=ft.Text("5D")),
            ft.Segment(value="1mo", label=ft.Text("1M")),
            ft.Segment(value="3mo", label=ft.Text("3M")),
            ft.Segment(value="6mo", label=ft.Text("6M")),
            ft.Segment(value="1y",  label=ft.Text("1A")),
            ft.Segment(value="max", label=ft.Text("MAX")),
        ],
        on_change=lambda e: asyncio.ensure_future(
            executar_calculo(novo_periodo=list(e.control.selected)[0])
        ),
    )

    # ── search bar ────────────────────────────────────────────────────────────
    tarefa_busca = None

    async def fechar_sugestao(e):
        # Quando o usuário clica em uma sugestão do auto-complete
        tk = e.control.data
        await pesquisa.close_view(tk)
        page.update()
        await executar_calculo(novo_ticker=tk)

    async def abrir_pesquisa(e):
        await e.control.open_view()

    def buscar_sugestoes_yahoo(query):
        """Consome um endpoint não documentado do Yahoo Finance para buscar auto-completes."""
        try:
            r = requests.get(
                f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0",
                headers={'User-Agent': 'Mozilla/5.0'}, timeout=3,
            )
            return r.json().get('quotes', [])
        except:
            return []

    async def atualizar_sugestoes(e):
        """Gerencia o auto-complete assíncrono com 'debounce' (cancelando buscas antigas)."""
        nonlocal tarefa_busca
        texto = e.control.value.lower()
        if len(texto) < 2:
            pesquisa.controls = []
            page.update()
            return
        
        # Cancela a tarefa anterior se o usuário continuou digitando rápido
        if tarefa_busca and not tarefa_busca.done():
            tarefa_busca.cancel()
        tarefa_busca = asyncio.create_task(_busca(texto))

    async def _busca(texto):
        """Realiza efetivamente a requisição HTTP após uma pequena pausa (debounce)."""
        try:
            await asyncio.sleep(0.5) # Espera 0.5s para evitar de spammar a API enquanto digita
            resultados = await asyncio.to_thread(buscar_sugestoes_yahoo, texto)
            
            # Constrói as list tiles com os resultados retornados
            pesquisa.controls = [
                ft.ListTile(
                    title=ft.Text(it.get('symbol',''), weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(it.get('shortname', it.get('longname', it.get('symbol','')))),
                    leading=ft.Icon(ft.Icons.SHOW_CHART, size=30, color=ft.Colors.BLUE_400),
                    data=it.get('symbol',''),
                    on_click=fechar_sugestao,
                ) for it in resultados
            ]
            page.update()
        except asyncio.CancelledError:
            pass # Busca foi cancelada intencionalmente

    async def enviar_pesquisa(e):
        # Quando o usuário aperta 'Enter'
        await executar_calculo(novo_ticker=e.control.value)

    pesquisa = ft.SearchBar(
        width=500,
        bar_hint_text="Pesquise o código da empresa...",
        view_hint_text="Buscando na rede...",
        on_change=atualizar_sugestoes,
        on_submit=enviar_pesquisa,
        on_tap=abrir_pesquisa,
    )

    # ── layout ────────────────────────────────────────────────────────────────
    # Adiciona todos os componentes à página do aplicativo Flet
    page.add(
        ft.Row(
            controls=[
                ft.Container(width=40),
                ft.Container(content=titulo, expand=True, alignment=ft.Alignment.CENTER),
                botao_tema,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        pesquisa,
        linha_historico,
        sk_row1,
        sk_row2,
        ft.Container(height=10),
        sk_grafico,
        painel_dados,
    )

ft.run(main)
# ft.run(main, view=ft.AppView.WEB_BROWSER)