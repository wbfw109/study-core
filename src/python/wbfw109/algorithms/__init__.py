"""
migrations guides from python >= 3.9 to python 3.7  (tested by pyenv 3.7.15)
    - changed and added contents
        from typing import Literal, Final: python >= 3.8
        Type hint of nested list: python >= 3.9  (error: object is not subscriptable)
        Assign Expressions(:=): python >= 3.8
    - but If possible, running immediately in my computer is effective rather than cloud by downloading test dataset with sample.
    The worst part is when you pass the sample test but fail the test case.
    - use VS code extension: replacerules.rulesets": {"Remove Type hint"} ...


* Complexity
    affected including: sorted()
* for Faster coding
    - Debugging: ipython, exit(), pprint, assert statement
    - not use @property, exception because of time save
    - use (dataclasses | NamedTuple)
    
        
📰~
* Tip
    - 보고서 가능하면 작은 단위부터 변수화하고 더 큰 단위가 있으면 변수화하자.
    - 기준을 잘 잡고, 가능한 알고리즘들 중 기회비용을 비교한다.
interactive 몇개 문b제 풀어보고, ProblemSolution 추상화클래스 상속받아서 작성 필요
★ 알고리즘..대회말고 따로 자료구조만들어서 테스트할것.. 자료구조 이론별로 패키지 정리? number_theory, graph_theory 문제 보고 자료구조 뭐쓸지 떠올려야함.
asyncio 테스트코드. 뭐랑? 리액트 타입스크립트?
비동기 함수 - 콜백. 비동기 함수가 리턴값을 받아 왔을떄 받아오는 값.? 이라고 정의하면 되나.
Flask, Django 할 떄 logging 사용하기.
이 프로젝트 안에서 Flask Django 를 공존할 수 있나? 이거 해보고 CI / CD 적용해보기
🔍 마이크로서비스 단위? 쿠버네티스 쓰고, Docker 는 교체해야하나?
2022 oci docker vs Buildah, Podman, Skopeo`
🔍 Flask, Django 용도 정하기
    Flask <-> AI 통신.
    Django doesn’t allow individual processes to handle multiple requests simultaneously.
    Developers have to come up with ways to make individual processes handle multiple requests
    - Flask 에서 해야하는, Django 에 내장되어잇던 것을들 해야 하는데 그 처리할 목록을 모르기 떄문에 Django 부터 하는게 낫나?
    
https://stateofjs.com/en-us/

대부분의 AI 모델은 플라스크와 어울린다고 한다.
TpeSCript 프로젝트?

pylint https://google.github.io/styleguide/pyguide.html 필요할까?
FastAPI?


"""
