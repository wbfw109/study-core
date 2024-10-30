// https://www.aiotkorea.or.kr/2024/kor/about/exhibitor.asp
const joinCompany = document.getElementById('joinCompany');

if (joinCompany) {
  const listItems = joinCompany.querySelectorAll(':scope > li');

  listItems.forEach(li => {
    const nameElement = li.querySelector(':scope > h4');
    const anchorElement = li.querySelector(':scope > a > dl > dd > p');
    const dlAnchorElement = li.querySelector(':scope > dl > a');

    const name = nameElement ? nameElement.innerText : 'No name';
    const text = anchorElement ? anchorElement.innerText : 'No text';
    const href = dlAnchorElement ? dlAnchorElement.getAttribute('href') : 'No href';

    console.log(`⚓ ${name} ; ${href}\n  ${text}`);
  });
} else {
  console.log('id가 "joinCompany"인 요소를 찾을 수 없습니다.');
}
