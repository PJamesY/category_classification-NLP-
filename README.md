# topic_modeling

##  1. 프로젝트 

- 네이버 지식인의 질문을 카테고리화 하기 위한 모델 제작

  -  [‘교육/학문’ , ‘컴퓨터/통신’, ‘엔터테인먼트’, ‘생활/경제’, ‘여행’, ‘스포츠’, ‘쇼핑’]

## 2. 사용한 데이터 

- training data : 한겨레 카테고리별 기사 데이터

  - 제가 training data를 기사 데이터로 쓴 이유는 기존의 네이버 지식인 질문 데이터는 문법, 띄어쓰기, 특수문자 등 데이터 전처리에 있어서 어려움이 있어 한겨레 기사 데이터를 training data로 하기로 하였습니다

  - 크롤링을 통하여 데이터를 얻었습니다 : www.hani.co.kr/

##  3. 모델링 방법

- LDA(Latent DIrichlet Allocation)

- Word Embedding / RNN

- Doc3vec