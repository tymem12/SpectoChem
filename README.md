# Graph-JEPA

Adaptation of [I-JEPA](https://arxiv.org/abs/2301.08243) method to graph domain.

## Installation
1. CPU environment
    ```shell
    make install_cpu
    ```
2. GPU environment
    ```shell
    make install_gpu
    ```


## 4. Outputy

* **TMQM_SPECTO_BINARY** – binarna klasyfikacja cząsteczek.
  Jeśli w zdefiniowanym zakresie długości fali istnieje przynajmniej jedna lambda o wartości oscylacji *f* powyżej ustalonego progu, cząsteczka jest uznawana za pozytywną; w przeciwnym wypadku – negatywną.

* **TMQM_SPECTO_LAMBDA_BINARY** – przewidywanie jedynie lambd.
  Jeśli istnieje przynajmniej jedna lambda o wartości oscylacji *f* powyżej progu, model powinien przewidzieć tę lub te lambdy.

* **TMQM_SPECTO_PAIRS** – przewidywanie par: (lambda + oscylacja).
  Analogicznie jak wyżej: jeśli istnieje lambda o wartości *f* powyżej progu, model przewiduje odpowiednie lambdy oraz wartości oscylacji.

* **TMQM_SPECTO_VECTOR** – przewidywanie wektora obejmującego cały przedział widzialnych długości fal.
  Każdy element wektora reprezentuje wartość oscylacji dla danej lambdy.

---

## Filtracja i konfiguracja

Każdy output zawiera następujące pola:

```yaml
additional_loading_params: 
  prediction_type: binary_classification
  num_states: 1
  vis_range: [380, 650]
  block_3_only: True
  filter_type: all_samples 
  filter_f_value: -1
  min_f_value: 0.001
  sort_by_max_f: false
  standarize_lambda: false
  standarize_f: false
```

### Objaśnienia parametrów

* **prediction_type: binary_classification**
  Typ przewidywania. Nie zmieniamy tej wartości.

* **num_states: 1**
  Liczba przewidywanych wartości (niezmienna dla `binary_classification`).
  Dla innych outputów określa liczbę przewidywanych lambd lub par lambda–f.
  Chemicy prosili o przewidywanie 4 lub 10 stanów, lecz większa liczba utrudnia modelowanie ze względu na liczne zera w wartościach *f*.
  Dla outputu typu *pairs*: rozmiar wyjścia to `2 * num_states`.

---

### Filtrowanie (`filter_type`, `filter_f_value`, `min_f_value`)

* **filter_type: all_samples**
  Alternatywa: `one_visible_lambda`.
  Parametr definiujący, które cząsteczki i lambdy zostają zachowane po filtracji.

  * **all_samples**
    Wybiera cząsteczki i ich `num_states` lambd/par, dla których `f > filter_f_value`.

    * `filter_f_value < 0` → zwraca wszystkie cząsteczki
    * `filter_f_value = 0` → usuwa lambda z `f = 0`

  * **one_visible_lambda**
    Wybiera cząsteczki i ich lambdy/pary z `f > filter_f_value`, oraz dodatkowo wymaga, aby w zakresie widzialnym istniała co najmniej jedna lambda o wartości `f ≥ min_f_value`.

* **filter_f_value: -1**
  Threshold filtrowania wartości *f*.

  * `< 0` → pozostawia również zera
  * `0` → usuwa tylko zera
  * `> 0` → usuwa wszystkie wartości `f` mniejsze od progu

* **min_f_value: 0.001**
  Próg oscylacji wykorzystywany:

  * do binarnej klasyfikacji (czy wartość w zakresie > próg),
  * w filtracji typu `one_visible_lambda` (czy istnieje choć jedna sensowna lambda).

---

### Pozostałe parametry

* **vis_range: [380, 650]**
  Definicja zakresu widzialnego.

* **block_3_only: True**
  Uwzględnia wyłącznie cząsteczki z bloku 3.

* **sort_by_max_f: false**
  Określa sposób sortowania zwracanych lambd:

  * `false` → sortowanie po długości fali
  * `true` → sortowanie po wartości oscylacji

* **standarize_lambda: false**
  Standardyzacja długości fali (stosowana tylko dla niektórych outputów).

* **standarize_f: false**
  Standardyzacja wartości oscylacji (analogicznie — zależna od outputu).

# UMA

## Accessing the model

### Requesting the access
1. Go to https://huggingface.co/facebook/UMA, fill out the model access form and send the request

1. Go to https://huggingface.co/settings/gated-repos and wait for the request status to change from PENDING to ACCEPTED

### Accessing the model in scripts
1. In order to run inference using the model, you need to authenticate your huggingface account that has access to the UMA repo, in one of two ways:

- one time authentication using `huggingface-cli` in the terminal:
```base
huggingface-cli login
```

- specifying a huggingface access token (created at https://huggingface.co/settings/tokens) every time as an environment variable when running an UMA script - either in the command line:
```bash
HF_TOKEN=... python script.py
```
or in the Python script itself:
```python
import os
os.environ["HF_TOKEN"] = "..."
```
