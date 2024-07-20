# Projeto de Reconhecimento de Gestos para Traduzir Libras

Este projeto implementa um sistema de reconhecimento de gestos utilizando a biblioteca PyTorch. O sistema pode ser treinado para reconhecer diferentes gestos capturados através de uma webcam e traduzir para Libras (Língua Brasileira de Sinais).

## Requisitos

- Python 3.7 ou superior
- Pip
- OpenCV
- PyTorch
- Torchvision

## Instalação

1. **Clone o repositório:**

    ```sh
    git clone https://github.com/jessicagasque/visao-computacional/libras-translate.git
    cd libras-translate
    ```

2. **Crie e ative um ambiente virtual:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate  
    No Windows: venv\Scripts\activate
    ```

3. **Instale as dependências:**

    ```sh
    pip install opencv-python-headless torch torchvision
    ```

## Estrutura de Diretórios

Certifique-se de que a estrutura de diretórios para os dados de treinamento e teste esteja correta:


    dataset/
    train/
        gesto1/
            img1.jpg
            img2.jpg
            ...
        gesto2/
            img1.jpg
            img2.jpg
            ...
        gesto3/
            img1.jpg
            img2.jpg
            ...
    test/
        gesto1/
            img1.jpg
            img2.jpg
            ...
        gesto2/
            img1.jpg
            img2.jpg
            ...
        gesto3/
            img1.jpg
            img2.jpg
            ...
    
**Passo 1: Verificar se há Imagens nos Diretórios**

Certifique-se de que há imagens nos diretórios dataset/train/gesto1, dataset/train/gesto2, dataset/train/gesto3, e assim por diante.

**Passo 2: Capturar Imagens (se necessário)**

Se ainda não tiver capturado imagens, use o script capture_images.py para adicionar imagens aos diretórios. Certifique-se de que está capturando imagens com extensões suportadas (.jpg, .jpeg, .png).

Execute o script:

    python3 capture_images.py

<ul>Pressione 'c' para capturar uma imagem.</ul>
<ul>Pressione 'n' para mudar para o próximo gesto.</ul>
<ul>Pressione 'q' para sair.</ul>

**Detecção de Mãos**

Use o script hand_detection.py para detectar mãos nas imagens capturadas e extrair as regiões de interesse. Este passo é opcional, mas recomendado para melhorar a precisão do modelo.

**Treinamento do Modelo**

Após capturar as imagens, use o script gesture_recognition.py para treinar o modelo.
Execute o script:

    python3 gesture_recognition.py

**Reconhecimento de Gestos em Tempo Real**

Use o script real_time_gesture_recognition.py para reconhecer gestos em tempo real através da webcam.
Execute o script:

    
    python3 real_time_gesture_recognition.py
    
### Conclusão

Siga as etapas acima para capturar dados, treinar o modelo e executar a detecção de gestos em tempo real. 
Se desejar contribuir para o projeto, sinta-se a vontade para abrir um pull request ou relatar problemas através da seção de issues do repositório.

### Autor

    jessicagasque


