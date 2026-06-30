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

function setDropdownOpen(dropdown, isOpen) {
  if (!dropdown) return;
  var button = dropdown.querySelector('.nav-dropdown-toggle');
  var submenu = dropdown.querySelector('.nav-submenu');
  if (!button || !submenu) return;
  dropdown.classList.toggle('open', isOpen);
  button.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  submenu.hidden = !isOpen;
}

function closeAllDropdowns(except) {
  document.querySelectorAll('.nav-dropdown').forEach(function (dropdown) {
    if (dropdown !== except) setDropdownOpen(dropdown, false);
  });
}

var menuButton = document.querySelector('.hamburger-btn');
var mobileMenuLinks = document.querySelectorAll('.mobile-overlay a');

if (menuButton) {
  menuButton.addEventListener('click', toggleMobileMenu);
}

document.querySelectorAll('.nav-dropdown').forEach(function (dropdown) {
  var button = dropdown.querySelector('.nav-dropdown-toggle');
  var submenu = dropdown.querySelector('.nav-submenu');

  if (button) {
    button.addEventListener('click', function (event) {
      event.stopPropagation();
      var isOpen = button.getAttribute('aria-expanded') !== 'true';
      closeAllDropdowns(dropdown);
      setDropdownOpen(dropdown, isOpen);
    });
  }

  dropdown.addEventListener('click', function (event) {
    event.stopPropagation();
  });

  if (submenu) {
    submenu.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        setDropdownOpen(dropdown, false);
      });
    });
  }
});

document.addEventListener('click', function () {
  closeAllDropdowns(null);
});

document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape') {
    document.querySelectorAll('.nav-dropdown.open').forEach(function (dropdown) {
      setDropdownOpen(dropdown, false);
      var button = dropdown.querySelector('.nav-dropdown-toggle');
      if (button) button.focus();
    });
  }
});

mobileMenuLinks.forEach(function (link) {
  link.addEventListener('click', closeMobileMenu);
});

addPagePrefetching();
