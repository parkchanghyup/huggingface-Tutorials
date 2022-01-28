```python
!pip install transformers
```


```python
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

```


```python
#모델 불러오기
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```


    Downloading:   0%|          | 0.00/443 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.25G [00:00<?, ?B/s]



```python
# 토크나이저 다운
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```


    Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/455k [00:00<?, ?B/s]


# 입력 전처리
1. 질문 시자 부분에 [CLS]토큰을 추가하고 질문과 단락 끝에 [SEP] 토큰을 추가한다.
2. 질문과 단락을 토큰화 한다.
3. segment_ids를 정의한다. 
    - 질문의 모든 토큰에 대해 0
    - 단락의 모든 토큰에 대해 1


```python
# 입력 전처리
question = "What is the immune system?"
paragraph = "The immune system is a system of many biological structures and processes within an organism that protects against disease. To function properly, an immune system must detect a wide variety of agents, known as pathogens, from viruses to parasitic worms, and distinguish them from the organism's own healthy tissue."
```


```python
question = '[CLS] '+question + '[SEP]'
paragraph = paragraph + '[SEP]'
```


```python
# 질문과 단락 토큰화
question_tokens = tokenizer.tokenize(question)
paragraph_tokens = tokenizer.tokenize(paragraph)
```


```python
question_tokens
```




    ['[CLS]', 'what', 'is', 'the', 'immune', 'system', '?', '[SEP]']




```python
# input_ids로 변환
tokens = question_tokens + paragraph_tokens
input_ids = tokenizer.convert_tokens_to_ids(tokens)
```


```python
input_ids
```




    [101,
     2054,
     2003,
     1996,
     11311,
     2291,
     1029,
     102,
     1996,
     11311,
     2291,
     2003,
     1037,
     2291,
     1997,
     2116,
     6897,
     5090,
     1998,
     6194,
     2306,
     2019,
     15923,
     2008,
     18227,
     2114,
     4295,
     1012,
     2000,
     3853,
     7919,
     1010,
     2019,
     11311,
     2291,
     2442,
     11487,
     1037,
     2898,
     3528,
     1997,
     6074,
     1010,
     2124,
     2004,
     26835,
     2015,
     1010,
     2013,
     18191,
     2000,
     26045,
     16253,
     1010,
     1998,
     10782,
     2068,
     2013,
     1996,
     15923,
     1005,
     1055,
     2219,
     7965,
     8153,
     1012,
     102]




```python
# segment_ids 정의 
# 질문의 모든 토큰에 대해 0으로, 단락의 모든 토큰에 대해 1로 정의
segment_ids = [0] * len(question_tokens)
segment_ids += [1] * len(paragraph_tokens)
```


```python
#input_ids 및 segment_ids -> 텐서로 변환

input_ids = torch.tensor([input_ids])
segment_ids = torch.tensor([segment_ids])
```

# 모델을 이용해 결과값 추론
1. 모든 토큰에 대한 시작 점수와 끝 점수를 반환하는 모델에 input_ids 및 segment_ids를 입력
2. 시작 점수가 가장 높은 토큰의 인덱스인 start_index와 끝 점수가 가장 높은 end_index 추출
3. 시작과 끝 인덱스 사이의 텍스트 범위를 답으로 출력.


```python
# 1. 모든 토큰에 대한 시작 점수와 끝 점수를 반환하는 모델에 input_ids 및 segment_ids를 입력
output= model(input_ids, token_type_ids = segment_ids)
```


```python
# 2. 시작 점수가 가장 높은 토큰의 인덱스인 start_index와 끝 점수가 가장 높은 end_index 추출
start_scores, end_scores = output[0],output[1]
```


```python
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
```


```python
# 3. 시작과 끝 인덱스 사이의 텍스트 범위를 답으로 출력.
print(' '.join(tokens[start_index : end_index+1]))
```

    a system of many biological structures and processes within an organism that protects against disease

