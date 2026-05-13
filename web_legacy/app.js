// Cargar productos de la API al iniciar
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('http://localhost:8000/products');
        const data = await res.json();
        
        const container = document.getElementById('products-container');
        container.innerHTML = ''; // Limpiar mensaje de carga
        
        if(data.products && data.products.length > 0) {
            data.products.forEach(prod => {
                const id = 'prod_' + prod.replace(/\s+/g, '_');
                container.innerHTML += `
                    <div>
                        <input type="checkbox" id="${id}" value="${prod}" class="product-checkbox">
                        <label for="${id}" class="product-label">${prod}</label>
                    </div>
                `;
            });
        } else {
            container.innerHTML = '<p>No se encontraron productos. ¿Está entrenado el modelo?</p>';
        }
        
        // Limitar la selección a máximo 5 productos
        const checkboxes = document.querySelectorAll('.product-checkbox');
        checkboxes.forEach(cb => {
            cb.addEventListener('change', function() {
                const checkedCount = document.querySelectorAll('.product-checkbox:checked').length;
                if (checkedCount > 5) {
                    this.checked = false;
                    alert('Solo puedes seleccionar un máximo de 5 productos para la canasta.');
                }
            });
        });

    } catch (e) {
        document.getElementById('products-container').innerHTML = '<p style="color:#ef4444">Error conectando a la API. Verifica que <b>server.py</b> esté ejecutándose.</p>';
    }
});

// === Configuración de la Animación de Red Neuronal ===
const animationArea = document.getElementById('animationArea');
const numInputs = 5; // Nodos de entrada visuales
const inputNodes = [];
const lines = [];

// Calculamos centro para que sea responsive visualmente
const areaWidth = 500;
const areaHeight = 150;

// Dibujar nodos de entrada (izquierda)
for(let i=0; i<numInputs; i++) {
    let node = document.createElement('div');
    node.className = 'node';
    node.style.left = '40px';
    node.style.top = (20 + i*25) + 'px';
    inputNodes.push(node);
    animationArea.appendChild(node);
    
    // Linea hacia la salida (derecha)
    let line = document.createElement('div');
    line.className = 'line';
    line.style.left = '60px';
    line.style.top = (30 + i*25) + 'px';
    
    let destX = 400; // Nodo de salida
    let destY = 70;
    
    let dx = destX - 60;
    let dy = destY - (30 + i*25);
    let length = Math.sqrt(dx*dx + dy*dy);
    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
    
    line.style.width = length + 'px';
    line.style.transform = `rotate(${angle}deg)`;
    lines.push(line);
    animationArea.appendChild(line);
}

// Dibujar nodo de salida (Perceptrón Central)
const outputNode = document.createElement('div');
outputNode.className = 'node';
outputNode.style.left = '400px';
outputNode.style.top = '60px';
outputNode.style.width = '30px';
outputNode.style.height = '30px';
animationArea.appendChild(outputNode);

// === Evento de Submit del Formulario ===
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Recolectar checkboxes seleccionados (El carrito)
    const checkboxes = document.querySelectorAll('.product-checkbox:checked');
    const cart = Array.from(checkboxes).map(cb => cb.value);

    if (cart.length === 0) {
        alert('Por favor selecciona al menos un producto para armar el carrito.');
        return;
    }

    // Preparar UI para animación
    document.getElementById('predictForm').style.display = 'none';
    document.getElementById('result').style.display = 'none';
    animationArea.style.display = 'block';
    
    // Resetear colores de la animación
    inputNodes.forEach(n => n.classList.remove('active'));
    lines.forEach(l => l.classList.remove('active'));
    outputNode.classList.remove('output-active');

    // Efecto de propagación (Forward Pass Visual)
    await new Promise(r => setTimeout(r, 200));
    inputNodes.forEach((n, i) => {
        // Activamos algunos nodos al azar simulando los productos de entrada
        if(Math.random() > 0.2) {
            setTimeout(() => {
                n.classList.add('active');
                lines[i].classList.add('active');
            }, i * 150);
        }
    });

    try {
        const modelChoice = document.getElementById('model_choice').value;

        // Enviar a la API Python
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cart: cart, model_choice: modelChoice })
        });

        const result = await response.json();

        // Sincronizar llegada de datos con el nodo de salida visualmente
        setTimeout(() => {
            outputNode.classList.add('output-active');
        }, 800);

        // Mostrar resultado real después de un pequeño delay escénico
        setTimeout(() => {
            animationArea.style.display = 'none';
            document.getElementById('predictForm').style.display = 'block';
            
            if (result.status === 'success') {
                document.getElementById('result').style.display = 'block';
                document.getElementById('recommended_product').innerText = result.recommended_product;
                document.getElementById('recommended_category').innerText = `Categoría: ${result.recommended_category}`;
                
                // Desplazar hacia abajo para ver el resultado
                document.getElementById('result').scrollIntoView({behavior: 'smooth'});
            } else {
                alert(result.detail || 'Error en la predicción');
            }
        }, 1800);

    } catch (error) {
        console.error('Error al consultar la API:', error);
        animationArea.style.display = 'none';
        document.getElementById('predictForm').style.display = 'block';
        alert('Hubo un error al conectar con la API. Asegúrate de que server.py esté corriendo.');
    }
});
