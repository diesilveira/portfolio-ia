# 💼 Portfolio de Diego Silveira

Portfolio profesional desarrollado con **MkDocs Material** que documenta mi evolución como desarrollador de software, proyectos realizados y aprendizajes continuos.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Diego%20Silveira-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/diego-silveira-uy/)
[![Portfolio](https://img.shields.io/badge/Portfolio-En%20Línea-green?style=flat-square&logo=github)](https://diego-silveira-uy.github.io/portfolio-ia/)

## 🚀 Acerca del Portfolio

Este sitio web sirve como:
- **Documentación profesional** de proyectos y experiencias
- **Bitácora de aprendizaje** con reflexiones y próximos pasos
- **Showcase técnico** de habilidades y tecnologías
- **Centro de recursos** con herramientas y referencias útiles

### 📂 Estructura del Contenido

```
docs/
├── index.md              # Página principal
├── acerca.md             # Perfil profesional y habilidades
├── recursos.md           # Herramientas y referencias
└── portfolio/
    ├── index.md          # Índice de proyectos
    ├── plantilla.md      # Template para nuevos proyectos
    └── 01-primera-entrada.md  # Proyecto: Configuración del Portfolio
```

## 🛠️ Tecnologías Utilizadas

- **[MkDocs](https://mkdocs.org/)** - Generador de sitios estáticos
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Tema moderno y responsive
- **[GitHub Pages](https://pages.github.com/)** - Hosting gratuito
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automático

### Características Implementadas

- ✅ Tema responsive con modo oscuro/claro
- ✅ Navegación intuitiva con búsqueda avanzada
- ✅ Syntax highlighting para código
- ✅ Lightbox para imágenes
- ✅ Admoniciones y elementos interactivos
- ✅ Enlaces a redes sociales
- ✅ Actualización automática de fechas

## 🔧 Desarrollo Local

### Requisitos
- Python 3.8+
- pip

### Instalación y Ejecución
```bash
# Clonar el repositorio
git clone https://github.com/diesilveira/portfolio-ia.git
cd portfolio-ia

# Instalar dependencias
pip install -r requirements.txt

# Servir localmente
mkdocs serve
```

El sitio estará disponible en `http://localhost:8000`

### Comandos Útiles
```bash
# Verificar configuración
mkdocs config

# Build para producción
mkdocs build

# Deploy manual (opcional)
mkdocs gh-deploy
```

## 📝 Metodología de Documentación

Cada proyecto sigue una estructura consistente:

1. **Contexto** - Situación y motivación
2. **Objetivos** - Metas específicas y medibles
3. **Actividades** - Tareas con estimaciones de tiempo
4. **Desarrollo** - Proceso técnico y decisiones
5. **Evidencias** - Resultados tangibles y código
6. **Reflexión** - Aprendizajes y mejoras futuras

## 🚀 Despliegue

El sitio se despliega automáticamente a GitHub Pages mediante GitHub Actions:
- **Trigger**: Push a la rama `main`
- **Build**: `mkdocs build --strict`
- **Deploy**: Automático a `gh-pages`

## 📞 Contacto

- **LinkedIn**: [diego-silveira-uy](https://www.linkedin.com/in/diego-silveira-uy/)
- **GitHub**: [diesilveira](https://github.com/diesilveira)
- **Portfolio**: [En línea](https://diego-silveira-uy.github.io/portfolio-ia/)

---

*"La documentación es tan importante como el código"* - Documentar para aprender, compartir y crecer.
