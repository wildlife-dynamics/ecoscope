function toggleMobileMenu() {
  var overlay = document.getElementById('mobile-overlay');
  var hamburgerIcon = document.getElementById('hamburger-icon');
  var closeIcon = document.getElementById('close-icon');
  if (!overlay || !hamburgerIcon || !closeIcon || !menuButton) {
    return;
  }

  var isOpen = overlay.classList.contains('open');

  if (isOpen) {
    overlay.classList.remove('open');
    hamburgerIcon.style.display = '';
    closeIcon.style.display = 'none';
    menuButton.setAttribute('aria-label', 'Open menu');
  } else {
    overlay.classList.add('open');
    hamburgerIcon.style.display = 'none';
    closeIcon.style.display = 'block';
    menuButton.setAttribute('aria-label', 'Close menu');
  }
}

function closeMobileMenu() {
  var overlay = document.getElementById('mobile-overlay');
  var hamburgerIcon = document.getElementById('hamburger-icon');
  var closeIcon = document.getElementById('close-icon');
  if (!overlay || !hamburgerIcon || !closeIcon || !menuButton) {
    return;
  }

  overlay.classList.remove('open');
  hamburgerIcon.style.display = '';
  closeIcon.style.display = 'none';
  menuButton.setAttribute('aria-label', 'Open menu');
}

var prefetchedPages = {};

function hasPrefetchLink(href) {
  return Array.prototype.some.call(document.querySelectorAll('link[rel="prefetch"]'), function (link) {
    return link.href.split('#')[0] === href;
  });
}

function prefetchSameSitePage(href) {
  var url;

  if (!href) {
    return;
  }

  try {
    url = new URL(href, window.location.href);
  } catch (error) {
    return;
  }

  if (url.origin !== window.location.origin) {
    return;
  }

  var prefetchHref = url.href.split('#')[0];

  if (prefetchedPages[prefetchHref] || hasPrefetchLink(prefetchHref)) {
    prefetchedPages[prefetchHref] = true;
    return;
  }

  var link = document.createElement('link');
  link.rel = 'prefetch';
  link.as = 'document';
  link.href = prefetchHref;
  document.head.appendChild(link);
  prefetchedPages[prefetchHref] = true;
}

function addPagePrefetching() {
  document.querySelectorAll('a[href]').forEach(function (link) {
    var href = link.getAttribute('href');
    var prefetch = function () {
      prefetchSameSitePage(href);
    };

    link.addEventListener('mouseenter', prefetch, { once: true });
    link.addEventListener('focus', prefetch, { once: true });
    link.addEventListener('touchstart', prefetch, { once: true, passive: true });
  });
}

function setDevelopDropdownOpen(isOpen) {
  if (!developDropdown || !developDropdownButton || !developSubmenu) {
    return;
  }

  developDropdown.classList.toggle('open', isOpen);
  developDropdownButton.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  developSubmenu.hidden = !isOpen;
}

var menuButton = document.querySelector('.hamburger-btn');
var mobileMenuLinks = document.querySelectorAll('.mobile-overlay a');
var developDropdown = document.querySelector('.nav-dropdown');
var developDropdownButton = document.querySelector('.nav-dropdown-toggle');
var developSubmenu = document.getElementById('develop-submenu');

if (menuButton) {
  menuButton.addEventListener('click', toggleMobileMenu);
}

if (developDropdownButton) {
  developDropdownButton.addEventListener('click', function (event) {
    event.stopPropagation();
    setDevelopDropdownOpen(developDropdownButton.getAttribute('aria-expanded') !== 'true');
  });
}

if (developDropdown) {
  developDropdown.addEventListener('click', function (event) {
    event.stopPropagation();
  });
}

if (developSubmenu) {
  developSubmenu.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () {
      setDevelopDropdownOpen(false);
    });
  });
}

document.addEventListener('click', function () {
  setDevelopDropdownOpen(false);
});

document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape' && developDropdownButton && developDropdownButton.getAttribute('aria-expanded') === 'true') {
    setDevelopDropdownOpen(false);
    developDropdownButton.focus();
  }
});

mobileMenuLinks.forEach(function (link) {
  link.addEventListener('click', closeMobileMenu);
});

addPagePrefetching();
