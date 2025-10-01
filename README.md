# CNN Audio Classifier
## Kuvaus
Tässä projektissa koulutin ja otin käyttöön Convolutional Neural Network (CNN) -pohjaisen mallin, joka tunnistaa ääniavainsanoja.
Malli käyttää syötteenä äänen spektrikuvaa (spectrogram) ja pystyy luokittelemaan ääniä eri avainsanoihin.

Projektissa on myös käyttöliittymä, joka hyödyntää mallia reaaliaikaiseen tunnistukseen tai tallennettujen äänitiedostojen analysointiin. Käyttöliittymässä on toteutettu itse koneoppimismallin tasot, jotka käyttävät koulutetun mallin painoja.

## Ominaisuudet
- Ääniavainsanojen tunnistus reaaliajassa ja tallennetuista äänistä.
- CNN-arkkitehtuuri spektrikuvien analysointiin.
- Täysin toimiva käyttöliittymä koulutetun mallin hyödyntämiseen.

## Mallin arkkitehtuuri
Koneoppimismalli koostuu seuraavista vaiheista:

- **Spectrogram** – muuttaa yksikanavaisen äänisignaalin aika ja taajuus tason tensoriksi
- **Resize** – syötesignaalin koon muuttaminen
- **Normalization** – arvot skaalataan sopivalle välille
- **Convolution 1** – ensimmäinen konvoluutiokerros
- **Convolution 2** – toinen konvoluutiokerros
- **Flatten** – monidimensioinen data yhdistetään vektoriksi
- **Dense** – tiheä kerros (fully connected)
- **Dense** – lopullinen luokittelukerros
<img width="625" height="209" alt="better_model" src="https://github.com/user-attachments/assets/07bdbbe5-d673-4e4c-9611-230d2a2d7291" />

> **Huom:** Mallin kaikki tasot on koodattu itse lopulliseen käyttökoodiin. Seuraavassa osiossa avaan näiden tasojen toiminnan periaatteet ja käytön.

## Spectrogram taso
Spectrogram on siis lyhyen ajan fourier muunnos, joka muuttaa yksiulotteisesta aikadatan jaksosta kaksiulotteiseksi tensoriksi: aika × taajuus.

- X-akseli (vaaka): aika (esim. sekunnit tai näytteenottopisteet)
- Y-akseli (pysty): taajuus (Hz)
- Väriarvo: amplitudi tai teho tietyllä taajuudella tietyllä hetkellä

**Esimerkkikuva**
- Tumma/lämmin väri tarkoittaa korkeaa amplitudia (ääntä) kyseisellä taajuudella ja hetkellä.
- Kylmempi väri tarkoittaa vähäisempää äänen voimakkuutta.



<img width="668" height="453" alt="image" src="https://github.com/user-attachments/assets/3227c050-8e08-4f8d-8110-12b5889dee48" />

## Resize taso
muuttaa syötesignaalin spektrin koon sopivaksi CNN:n syötteeksi. Tämä varmistaa, että kaikki näytteet ovat yhtenäisiä mallin käsittelyä varten.

**Toimintaperiaate**

Käytetty algoritmi on **bilineaarinen interpolaatio** jossa input tensorista / kuvasta otetaan tasasin välein haluttu määrä arvoja ja lasketaan niitä ympäröivien pikseleiden keskiarvo.
- Säilyttää spektrin piirteet suhteellisen hyvin  
- Varmistaa, että eri pituisten ääninäytteiden spektrit saadaan yhtenäiseen kokoon  
- Valittu yksinkertaisuuden ja tehokkuuden vuoksi, eikä se vaadi ulkopuolisia kirjastoja  
<img width="800" height="278" alt="image" src="https://github.com/user-attachments/assets/9f658a9e-253d-465c-9326-f9b067493f50" />

## Normalization taso
Normalization-taso skaalaa spektrikuvan arvot sopivalle välille **0–1**

**Miksi teen näin?**
- Malli oppii paremmin kun spectrikuvien arvot on rajattu samalle alueelle
- Estää suurten arvojen dominointia opetusvaiheessa
- Parantaa vakautta
  
**Toimintaperiaate**
  Käytetty kaava on z-score normalization, joka standardisoi syötteen arvot niin, että ne keskitetään nollan ympärille ja niiden hajonta on yhden luokkaa. Tämä auttaa mallia oppimaan tehokkaammin ja vakaammin.
- Lasketaan spektrin minimi- ja maksimiarvot.
- Skaalataan jokainen arvo kaavalla


**Kaava**
  
<img width="293" height="135" alt="image" src="https://github.com/user-attachments/assets/58f8fb5c-c70e-42d9-b187-81f35efa0439" />

Missä:

- x = alkuperäinen arvo
- μ = syötteen keskiarvo
- σ² = syötteen varianssi
- ϵ = 1×10⁻⁷, estää nollalla jakamisen

## Convolution taso
![convolution-2](https://github.com/user-attachments/assets/c251bc7b-afb5-4f81-a296-1e90da849928)

Tämä taso suorittaa konvoluution syötetensorille. Konvoluutio on matemaattinen operaatio, jossa pieni suodatin/kernel liukuu syötedatan yli ja yhdistää arvoja pisteittäin. Konvoluution tarkoitus on tunnistaa piirteitä datasta. Esimerkiksi tässä projektissa malli etsii audiodatasta taajuuskuvioita tai puheenrytmejä. Projektissani malli käyttää konvoluutiota oppiakseen äänikomentojen ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'] piirteet.

**Toimintaperiaate**
1. Suodatin (esimerkiksi 3x3 matriisi) menee syöte tensorin läpi
2. Jokaisella kerralla lasketaan painotettu summa suodattimen ja vastaavan kuvan alueen arvoista
3. Tulos muodostaa uuden tensorin

**Huom!** voidaan toistaa useilla kerroksilla, jotta malli oppii erikoisempia piirteitä

**Konvoluution laskukaavoja**
Miten yksi filtteri laskee yhden konvoluutio pisteen
<img width="569" height="160" alt="image" src="https://github.com/user-attachments/assets/3f32c25d-e811-418c-a8b5-684b38974225" />

Missä:
y[i,j]       = ulostulon arvo kohdassa (i, j)
x[i+m,j+n]   = syötteen arvo kohdassa, jota kernel kattaa
w[m,n]       = kernelin (filterin) paino
b            = bias
K_H, K_W     = kernelin korkeus ja leveys

Miten ulostulo tensorin koko lasketaan

<img width="510" height="195" alt="image" src="https://github.com/user-attachments/assets/8a5ae4bd-53e5-444a-8391-d87ef176ecf3" />



**Esimerkki GIF** kuvaa miten konvoluutio tapahtuu kolmekanavaiseen RGB - kuvaan

![b1d5931b-9e37-42f9-8eae-02840dc968b2_960x540](https://github.com/user-attachments/assets/5b8549f8-460c-49fd-a023-0db7b98a16ff)

