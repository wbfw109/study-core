# Plan

---

Updated on 📅 2023-02-09 11:38:36

- [Plan](#plan)
  - [1. Plan](#1-plan)
    - [1.1. Motivation](#11-motivation)
    - [1.2. Requirements](#12-requirements)
      - [1.2.1. Functional Requirement](#121-functional-requirement)
      - [1.2.2. Non-Functional Requirement](#122-non-functional-requirement)
  - [2. Define](#2-define)
    - [2.1. Nodes](#21-nodes)
    - [2.2. Work breakdown structure](#22-work-breakdown-structure)

## 1. Plan

### 1.1. Motivation

- Services using Machine Learning have Similar form.
  - NoSQL DBMS for unstructured data and ElasticSearch for fast searching.
  - Apache Kafka for Data Streaming which uses Publish-Subscribe pattern.
  - Backend Server and, Web Frontend or Application.
  - These are maintained as Container and deployed to Cluster and Cloud Environment.

- I heard Push notification functionality of PWA will be supported on IOS from WWDC 2022 (already available in Android).  
  Also PassKeys technology for Authorization is evolving with support from MS, Apple, and Google.

- In web development, Jamstack's interest is rising (Backend for Frontend pattern).

- If target of AI service is E-Commerce, I could experience to implement functions:
  - Payment system in way of subscription or one-time with optional products.
  - Recommendation system based on Deep learning.
  - Notification function from PWA.

So I will try implementation systematically these schemes through GitOPs practices by representing UML form as possible.

&nbsp;

---

### 1.2. Requirements

#### 1.2.1. Functional Requirement

📰 To be determined

#### 1.2.2. Non-Functional Requirement

- To avoid overload and reduce response time, **Load balancing** is required in layers which larger data transport is occurred in such as in API, Database operations.
- The program must be able to operated on OS including Unix-like, Windows.

&nbsp;

---

## 2. Define

### 2.1. Nodes

1. **\<ML Model\>** ; pseudo unit which infers required Problems
2. Database
    1. **MongoDB** ; Document-based NoSQL DBMS
    2. **Mongo Express** ; Lightweight web-based administrative interface
3. **ElasticSearch** ; Searching engine. and components **ELK stack:**
    1. **Logstash** ; Data collection and log-parsing engine
    2. **Kibana** ; Analytics and visualization platform
    3. **Beats** ; Collection of lightweight data shippers
4. **Kafka** ; Distributed event store and stream processing with Publish-Subscribe Pattern
5. API
    1. **FastAPI** ; 🔐 [_MIT license_](https://github.com/tiangolo/fastapi/blob/master/LICENSE) with **Gunicorn** (using **uvicorn**)
    2. Protocol Buffers ; 🔐 [_MIT license_](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE)
    3. gRPC
6. proxy
   1. **NginX**
7. Frontend: **Svelte** (Typescript) with PWA
8. package type: Container
    1. container platform: **Docker**
    2. orchestration: **Kubernetes**
9. CI/CD
    1. GitHub Actions  <!-- , JenKins CI, Argo CD -->
9.Version Control: **GitHub**

### 2.2. Work breakdown structure

Omit implementation

- **\<ML Model\>** ; because of pseudo unit according to a Business model

📰 To be determined
