-- keep-html-table.lua  v2
--local function ensure_html_table_is_rawblock(text)
--  return text:gsub(
    --"(.-)(<%s*[Tt][Aa][Bb][Ll][Ee].->.-</[Tt][Aa][Bb][Ll][Ee]>)(.-)",
    --function (pre, tab, post)
--      return pre .. "\n\n" .. tab .. "\n\n" .. post
    --end)
--end

--function Para (el)
--  local new_content = ensure_html_table_is_rawblock(pandoc.utils.stringify(el))
--  if new_content:match("^%s*<%s*[Tt][Aa][Bb][Ll][Ee]") then
    --return pandoc.RawBlock('html', new_content)
  --end
--end

function RawBlock (el)
  if el.format:match 'html' and el.text:match '%<table' then
    return pandoc.read(el.text, el.format).blocks
  end
end