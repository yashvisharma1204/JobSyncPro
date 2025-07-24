
       
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

    
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (window.scrollY > 50) {
                navbar.style.background = 'rgb(11, 17, 30,0.08)';
                navbar.style.boxShadow = '0 4px 20px rgba(59, 130, 246, 0.1)';
            } else {
                navbar.style.background = 'rgb(11, 17, 30,0.95)';
                navbar.style.boxShadow = 'none';
            }
        });

       
        document.querySelectorAll('.bento-item, .process-card, .spec-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-4px)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
