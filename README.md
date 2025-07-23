# TechDepth - Advanced Tech Blog

A professional Jekyll-based tech blog focused on advanced programming concepts, featuring in-depth coverage of Golang, Java, AI, and Computer Science fundamentals.

## 🚀 Features

- **Professional Design**: Beautiful dark theme with animated elements
- **Jekyll-Powered**: Static site generation with GitHub Pages support
- **Category Organization**: Organized posts by Golang, Java, AI, and CS Concepts
- **SEO Optimized**: Proper meta tags, sitemap, and RSS feed
- **Mobile Responsive**: Works perfectly on all devices
- **Syntax Highlighting**: Code blocks with proper language support
- **Easy Post Creation**: Template-based workflow for quick publishing

## 📁 Project Structure

```
tech-blog/
├── _config.yml              # Jekyll configuration
├── _layouts/                 # Page templates
│   ├── default.html         # Base layout with navigation/footer
│   └── post.html            # Blog post layout
├── _posts/                   # Blog posts (Markdown)
│   ├── 2025-01-27-golang-garbage-collector-deep-dive.md
│   ├── 2025-01-26-java-g1-vs-zgc-garbage-collectors.md
│   ├── 2025-01-25-transformer-architecture-evolution.md
│   └── 2025-01-24-btree-database-indexing.md
├── _post_template.md         # Template for new posts
├── Gemfile                   # Ruby dependencies
├── index.html               # Homepage
└── README.md                # This file
```

## 🛠️ Getting Started

### Prerequisites

- Ruby 2.7 or higher
- Bundler gem
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tech-blog
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run locally**
   ```bash
   bundle exec jekyll serve
   ```

4. **View your site**
   Open http://localhost:4000 in your browser

### GitHub Pages Deployment

Your blog is already configured for GitHub Pages! Simply:

1. **Enable GitHub Pages**
   - Go to repository Settings
   - Scroll to Pages section
   - Set Source to "Deploy from a branch"
   - Select `main` branch
   - Click Save

2. **Your blog will be live at**
   ```
   https://yourusername.github.io/tech-blog
   ```

## ✍️ Creating New Posts

### Using the Template

1. **Copy the template**
   ```bash
   cp _post_template.md _posts/YYYY-MM-DD-your-post-title.md
   ```

2. **Fill in the frontmatter**
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: 2025-01-27 10:00:00 -0000
   categories: [golang]  # Choose: golang, java, ai, cs-concepts, optimization
   tags: [tag1, tag2, tag3]
   author: "TechDepth Team"
   reading_time: 8
   excerpt: "Brief description for previews"
   ---
   ```

3. **Write your content in Markdown**

### Post Guidelines

- **File naming**: `YYYY-MM-DD-post-title-with-hyphens.md`
- **Categories**: Choose from `golang`, `java`, `ai`, `cs-concepts`, `optimization`
- **Tags**: 3-5 relevant keywords, lowercase with hyphens
- **Reading time**: Estimate ~200 words per minute
- **Length**: 1500-3000 words for substantial content

### Code Examples

Use fenced code blocks with language specification:

```golang
func main() {
    fmt.Println("Hello, TechDepth!")
}
```

## 📝 Content Categories

### 🐹 Golang
- Concurrency patterns
- Memory management
- Performance optimization
- Language internals

### ☕ Java
- JVM internals
- Garbage collection
- Enterprise patterns
- Performance tuning

### 🤖 AI
- Machine learning algorithms
- Neural network architectures
- AI system design
- Latest research insights

### 🧠 CS Concepts
- Data structures
- Algorithms
- System design
- Theoretical foundations

## 🎨 Customization

### Updating Site Configuration

Edit `_config.yml` to customize:
- Site title and description
- Author information
- Social media links
- Analytics tracking

### Modifying the Design

- **Colors**: Edit CSS variables in `_layouts/default.html`
- **Layout**: Modify templates in `_layouts/`
- **Homepage**: Update `index.html`

### Adding New Categories

1. Add to `_config.yml`:
   ```yaml
   category_descriptions:
     your-category: "Description of your category"
   ```

2. Update navigation in `_layouts/default.html`

## 📊 Analytics & SEO

The blog is configured with:
- **SEO tags**: Automatic meta tags and Open Graph
- **Sitemap**: Auto-generated XML sitemap
- **RSS Feed**: Available at `/feed.xml`
- **Google Analytics**: Add your tracking ID to `_config.yml`

## 🚀 Performance Features

- **Syntax highlighting**: Rouge highlighter
- **Responsive images**: Automatic optimization
- **Fast loading**: Minimal dependencies
- **CDN ready**: Works with GitHub Pages CDN

## 📱 Mobile Optimization

The blog is fully responsive with:
- Mobile-first design
- Touch-friendly navigation
- Optimized reading experience
- Progressive enhancement

## 🔧 Troubleshooting

### Common Issues

**Build errors**:
```bash
bundle update
bundle exec jekyll build --verbose
```

**Gem conflicts**:
```bash
bundle clean --force
bundle install
```

**GitHub Pages build fails**:
- Check `_config.yml` syntax
- Ensure all posts have valid frontmatter
- Verify Gemfile compatibility

### Development Tips

- Use `bundle exec jekyll serve --drafts` to preview draft posts
- Add `--livereload` for automatic browser refresh
- Check `_site/` folder for generated output

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write your post following the guidelines
4. Submit a pull request

---

**Happy blogging!** 🚀

For questions or suggestions, feel free to open an issue or reach out to the TechDepth team.
# Last updated: Thu Jul 24 04:19:16 IST 2025
